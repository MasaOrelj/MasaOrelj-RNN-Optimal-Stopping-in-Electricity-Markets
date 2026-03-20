import time
import numpy as np
import torch

from optimal_stopping.algorithms.backward_induction import regression
from optimal_stopping.algorithms.backward_induction import backward_induction_pricer
from optimal_stopping.run import configs


class SwingRegressionPricer:
    """
    Base class for regression-based swing pricing.

    The backward induction is implemented here once.
    Subclasses only need to set self.regression.
    """

    def __init__(
        self,
        model,
        payoff,
        nb_epochs=None,
        nb_batches=None,
        train_ITM_only=True,
        use_payoff_as_input=False,
        use_spot_as_input=True,
        use_var=None,
        num_swings=1,
        exercise_dates=None,
        return_diagnostics=False,
        verbose=False,
        record_stopping_times=True,
    ):
        self.model = model
        self.payoff = payoff

        # kept for compatibility with runner signature
        self.nb_epochs = nb_epochs
        self.nb_batches = nb_batches

        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_spot_as_input = use_spot_as_input
        self.num_swings = num_swings
        self.exercise_dates = exercise_dates
        self.return_diagnostics = return_diagnostics
        self.verbose = verbose
        self.record_stopping_times = record_stopping_times
        self.all_exercise_dates = None

        if use_var is None:
            self.use_var = getattr(model, "return_var", False)
        else:
            self.use_var = use_var

        self.input_dim = backward_induction_pricer.compute_input_dim(
            model=self.model,
            use_var=self.use_var,
            use_payoff_as_input=self.use_payoff_as_input,
            use_spot_as_input=self.use_spot_as_input,
        )

        if self.input_dim == 0:
            raise ValueError(
                "At least one of spot, var, or payoff must be used as regression input."
            )

        self.regression = None
        self.last_diagnostics = None
        self.last_ex_dates = None
        self.split = None

    
    def _roll_forward_stopping_times(self,
    payoff_matrix,
    ex_dates,
    exercise_value_hat_store,
    wait_value_hat_store,
    split,):
        
        
        """
        Recover the realized swing exercise schedule on the evaluation paths only.

        This uses the pathwise fitted values computed during backward induction.
        It is valid because we roll forward on the same simulated paths.
        """
        nb_paths, nb_dates_plus_1 = payoff_matrix.shape
        eps = np.finfo(float).eps

        all_exercise_dates = np.zeros((nb_paths, nb_dates_plus_1), dtype=int)

        # optional compatibility object; first realized exercise date per path
        tau = np.full(nb_paths, fill_value=-1, dtype=int)

        rights_left = np.zeros(nb_paths, dtype=int)
        rights_left[split:] = self.num_swings

        eval_mask = np.zeros(nb_paths, dtype=bool)
        eval_mask[split:] = True

        for k, date in enumerate(ex_dates):
            active = eval_mask & (rights_left > 0)
            if not np.any(active):
                break

            immediate = payoff_matrix[:, date]
            exercise = np.zeros(nb_paths, dtype=bool)

            # terminal date: exercise if payoff is positive and rights remain
            if k == len(ex_dates) - 1:
                exercise = active & (immediate > eps)
            else:
                for r in range(1, self.num_swings + 1):
                    idx = active & (rights_left == r)
                    if not np.any(idx):
                        continue

                    ex_hat = exercise_value_hat_store[(k, r)]
                    wait_hat = wait_value_hat_store[(k, r)]

                    exercise[idx] = (
                        (immediate[idx] > eps)
                        & (ex_hat[idx] > wait_hat[idx])
                    )

            all_exercise_dates[exercise, date] = 1

            # first realized exercise date, for compatibility only
            first_hit = exercise & (tau == -1)
            tau[first_hit] = date

            rights_left[exercise] -= 1

        # paths that never exercised keep terminal date for compatibility
        tau[tau == -1] = ex_dates[-1]

        return all_exercise_dates, tau

    @staticmethod
    def _normalize_payoff_matrix(payoff_values):
        payoff_values = np.asarray(payoff_values)

        if payoff_values.ndim == 2:
            return payoff_values

        if payoff_values.ndim == 3 and payoff_values.shape[1] == 1:
            return payoff_values[:, 0, :]

        raise ValueError(
            "Payoff must return shape (nb_paths, nb_dates+1) "
            "or (nb_paths, 1, nb_dates+1)."
        )

    @staticmethod
    def _build_features(
        stock_paths,
        var_paths,
        payoff_matrix,
        date,
        use_var,
        use_payoff_as_input,
        use_spot_as_input,
    ):
        features = []

        if use_spot_as_input:
            features.append(stock_paths[:, :, date])

        if use_var:
            if var_paths is None:
                raise ValueError("use_var=True but model did not return var_paths.")
            features.append(var_paths[:, :, date])

        if use_payoff_as_input:
            features.append(payoff_matrix[:, date].reshape(-1, 1))

        if not features:
            raise ValueError(
                "At least one of spot, var, or payoff must be used as regression input."
            )

        return np.concatenate(features, axis=1)

    @staticmethod
    def _itm_sets(immediate_exercise_value, split, train_ITM_only):
        if train_ITM_only:
            in_the_money = np.where(immediate_exercise_value[:split] > 0.0)
            in_the_money_all = np.where(immediate_exercise_value > 0.0)
        else:
            in_the_money = np.where(immediate_exercise_value[:split] < np.inf)
            in_the_money_all = np.where(immediate_exercise_value < np.inf)

        return in_the_money, in_the_money_all

    def calculate_continuation_value(self, values, immediate_exercise_value, X_t):
        """
        Generic continuation estimator used by all regression-based swing methods.

        values: discounted next-step targets
        immediate_exercise_value: payoff at current date, used only for ITM filtering
        X_t: regression input at current date
        """
        in_the_money, in_the_money_all = self._itm_sets(
            immediate_exercise_value, self.split, self.train_ITM_only
        )

        continuation_values = np.zeros(X_t.shape[0])

        if len(in_the_money[0]) > 0 and len(in_the_money_all[0]) > 0:
            continuation_values[in_the_money_all[0]] = (
                self.regression.calculate_regression(
                    X_t, values, in_the_money, in_the_money_all
                )
            )
        return continuation_values

    def price(self, train_eval_split=2):
        if self.regression is None:
            raise ValueError("self.regression must be set in a subclass.")

        # 1. Generate paths
        t0 = time.time()
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
        stock_paths, var_paths = self.model.generate_paths()
        time_for_path_gen = time.time() - t0

        nb_paths, nb_stocks, nb_dates_plus_1 = stock_paths.shape

        # 2. Evaluate payoff
        payoff_values = self.payoff(stock_paths)
        payoff_matrix = self._normalize_payoff_matrix(payoff_values)

        if payoff_matrix.shape != (nb_paths, nb_dates_plus_1):
            raise ValueError(
                "Normalized payoff matrix must have shape (nb_paths, nb_dates+1)."
            )

        # 3. Split
        split = int(nb_paths / train_eval_split)
        self.split = split

        # 4. Exercise dates
        if self.exercise_dates is None:
            ex_dates = np.arange(1, nb_dates_plus_1, dtype=int)
        else:
            ex_dates = np.asarray(self.exercise_dates, dtype=int)

            if ex_dates.ndim != 1:
                raise ValueError("exercise_dates must be a 1D array of date indices.")
            if len(ex_dates) == 0:
                raise ValueError("exercise_dates cannot be empty.")
            if np.any(ex_dates <= 0):
                raise ValueError("exercise_dates must be strictly greater than 0.")
            if np.any(ex_dates >= nb_dates_plus_1):
                raise ValueError("exercise_dates must be <= nb_dates.")
            if np.any(np.diff(ex_dates) <= 0):
                raise ValueError("exercise_dates must be strictly increasing.")

        # 5. Terminal values
        terminal_date = ex_dates[-1]
        terminal_payoff = payoff_matrix[:, terminal_date]

        U = np.zeros((self.num_swings + 1, nb_paths))
        for r in range(1, self.num_swings + 1):
            U[r, :] = terminal_payoff

        continuation_ex_store = {}
        continuation_wait_store = {}
        exercise_decision_store = {}

        exercise_value_hat_store = {}
        wait_value_hat_store = {}

        result = {
            "stock_paths": stock_paths,
            "var_paths": var_paths,
            "payoff_matrix": payoff_matrix,
            "exercise_dates": ex_dates,
            "terminal_date": terminal_date,
            "terminal_payoff": terminal_payoff,
            "U_terminal": U.copy(),
            "split": split,
            "input_dim": self.input_dim,
        }

        if self.verbose:
            print(result)

        tau = np.full(nb_paths, fill_value=terminal_date, dtype=int)
        all_exercise_dates = np.zeros((nb_paths, nb_dates_plus_1), dtype=int)

        # 6. Full backward induction
        for k in range(len(ex_dates) - 2, -1, -1):
            date = ex_dates[k]
            next_date = ex_dates[k + 1]

            D = self.model.disc_factor(date, next_date)

            U_prev = U.copy()
            U = np.zeros_like(U_prev)

            r_min = max(self.num_swings - k, 0)

            X_t = self._build_features(
                stock_paths=stock_paths,
                var_paths=var_paths,
                payoff_matrix=payoff_matrix,
                date=date,
                use_var=self.use_var,
                use_payoff_as_input=self.use_payoff_as_input,
                use_spot_as_input=self.use_spot_as_input,
            )

            immediate_exercise_value = payoff_matrix[:, date]

            U[0, :] = 0.0

            for r in range(max(r_min, 1), self.num_swings + 1):
                # continuation if exercise now
                Y_ex = D * U_prev[r - 1, :]
                C_ex = self.calculate_continuation_value(
                    Y_ex, immediate_exercise_value, X_t
                )
                exercise_value_hat = immediate_exercise_value + C_ex

                # continuation if wait
                Y_wait = D * U_prev[r, :]
                C_wait = self.calculate_continuation_value(
                    Y_wait, immediate_exercise_value, X_t
                )

                if self.record_stopping_times:
                    exercise_value_hat_store[(k, r)] = exercise_value_hat.copy()
                    wait_value_hat_store[(k, r)] = C_wait.copy()

                exercise = (
                    (immediate_exercise_value > np.finfo(float).eps)
                    & (exercise_value_hat > C_wait)
                )

                U[r, ~exercise] = Y_wait[~exercise]
                U[r, exercise] = (
                    immediate_exercise_value[exercise]
                    + D * U_prev[r - 1, exercise]
                )

                if self.return_diagnostics:
                    continuation_ex_store[(k, r)] = C_ex.copy()
                    continuation_wait_store[(k, r)] = C_wait.copy()
                    exercise_decision_store[(k, r)] = exercise.copy()

        first_ex_date = ex_dates[0]
        D0 = self.model.disc_factor(0, first_ex_date)

        price_eval = D0 * np.mean(U[self.num_swings, split:])
        price_train = D0 * np.mean(U[self.num_swings, :split]) if split > 0 else np.nan

        self.last_ex_dates = tau

        if self.return_diagnostics:
            self.last_diagnostics = {
                "price_eval": price_eval,
                "price_train": price_train,
                "U0": U.copy(),
                "exercise_dates": ex_dates,
                "terminal_date": terminal_date,
                "first_ex_date": first_ex_date,
                "discount_to_first_ex_date": D0,
                "split": split,
                "input_dim": self.input_dim,
                "use_var": self.use_var,
                "use_spot_as_input": self.use_spot_as_input,
                "use_payoff_as_input": self.use_payoff_as_input,
                "stock_paths": stock_paths,
                "var_paths": var_paths,
                "payoff_matrix": payoff_matrix,
                "continuation_exercise": continuation_ex_store,
                "continuation_wait": continuation_wait_store,
                "exercise_decisions": exercise_decision_store,
            }
        else:
            self.last_diagnostics = None
        
        if self.record_stopping_times:
            all_exercise_dates, tau = self._roll_forward_stopping_times(
                payoff_matrix=payoff_matrix,
                ex_dates=ex_dates,
                exercise_value_hat_store=exercise_value_hat_store,
                wait_value_hat_store=wait_value_hat_store,
                split=split,
            )
            self.all_exercise_dates = all_exercise_dates
            self.last_ex_dates = tau
        else:
            self.all_exercise_dates = None
            self.last_ex_dates = None

        return price_eval, time_for_path_gen


class SwingLeastSquaresPricer(SwingRegressionPricer):
    def __init__(
        self,
        model,
        payoff,
        nb_epochs=None,
        nb_batches=None,
        train_ITM_only=True,
        use_payoff_as_input=False,
        use_spot_as_input=True,
        use_var=None,
        num_swings=1,
        exercise_dates=None,
        return_diagnostics=False,
        verbose=False,
        record_stopping_times=True, 
    ):
        super().__init__(
            model=model,
            payoff=payoff,
            nb_epochs=nb_epochs,
            nb_batches=nb_batches,
            train_ITM_only=train_ITM_only,
            use_payoff_as_input=use_payoff_as_input,
            use_spot_as_input=use_spot_as_input,
            use_var=use_var,
            num_swings=num_swings,
            exercise_dates=exercise_dates,
            return_diagnostics=return_diagnostics,
            verbose=verbose,
            record_stopping_times=record_stopping_times,
        )
        self.regression = regression.LeastSquares(self.input_dim)


class SwingReservoirLeastSquarePricerFast(SwingRegressionPricer):
    def __init__(self, model, payoff, hidden_size=10, factors=(1.,), nb_epochs=None,
                 nb_batches=None, train_ITM_only=True, use_payoff_as_input=False,
                 use_spot_as_input=True, use_var=None, num_swings=1,
                 exercise_dates=None, return_diagnostics=False, verbose=False, record_stopping_times=True):
        super().__init__(
            model, payoff, nb_epochs, nb_batches, train_ITM_only,
            use_payoff_as_input, use_spot_as_input, use_var,
            num_swings, exercise_dates, return_diagnostics, verbose, record_stopping_times
        )
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks
        self.regression = regression.ReservoirLeastSquares2(
            self.input_dim,
            hidden_size,
            activation=torch.nn.LeakyReLU(factors[0] / 2),
            factors=factors[1:],
        )
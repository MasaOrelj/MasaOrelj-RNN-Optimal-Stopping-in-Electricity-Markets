""" Underlying model of the stochastic processes that are used:
- Black Scholes
- Heston
- Fractional Brownian motion
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import scipy.special as scispe

import joblib


NB_JOBS_PATH_GEN = 1

class Model:
  def __init__(self, drift, dividend, volatility, spot, nb_stocks,
               nb_paths, nb_dates, maturity, name, **keywords):
    self.name = name
    self.drift = drift - dividend
    self.rate = drift
    self.dividend = dividend
    self.volatility = volatility
    self.spot = spot
    self.nb_stocks = nb_stocks
    self.nb_paths = nb_paths
    self.nb_dates = nb_dates
    self.maturity = maturity
    self.dt = self.maturity / self.nb_dates
    self.df = math.exp(-self.rate * self.dt)
    self.return_var = False
    self.var_dim = 0

  def disc_factor(self, date_begin, date_end):
    time = (date_end - date_begin) * self.dt
    return math.exp(-self.drift * time)

  def drift_fct(self, x, t):
    raise NotImplemented()

  def diffusion_fct(self, x, t, v=0):
    raise NotImplemented()

  def generate_one_path(self):
      raise NotImplemented()

  def generate_paths(self, nb_paths=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates+1) with prices."""
    nb_paths = nb_paths or self.nb_paths
    if NB_JOBS_PATH_GEN > 1:
        return np.array(
            joblib.Parallel(n_jobs=NB_JOBS_PATH_GEN, prefer="threads")(
                joblib.delayed(self.generate_one_path)()
                for i in range(nb_paths)))
    else:
        return np.array([self.generate_one_path() for i in range(nb_paths)]), \
               None


##############################
# Black Scholes
##############################

class BlackScholes(Model):
  def __init__(self, drift, volatility, nb_paths, nb_stocks, nb_dates, spot,
         maturity, dividend=0, **keywords):
    super(BlackScholes, self).__init__(
        drift=drift, dividend=dividend, volatility=volatility,
        nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
        spot=spot, maturity=maturity, name="BlackScholes")

  def drift_fct(self, x, t):
    del t
    return self.drift * x

  def diffusion_fct(self, x, t, v=0):
    del t
    return self.volatility * x

  def generate_paths(self, nb_paths=None, return_dW=False, dW=None, X0=None,
                     nb_dates=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
    nb_paths = nb_paths or self.nb_paths
    nb_dates = nb_dates or self.nb_dates
    spot_paths = np.empty((nb_paths, self.nb_stocks, nb_dates+1))
    if X0 is None:
        spot_paths[:, :, 0] = self.spot
    else:
        spot_paths[:, :, 0] = X0
    if dW is None:
        random_numbers = np.random.normal(
            0, 1, (nb_paths, self.nb_stocks, nb_dates))
        dW = random_numbers * np.sqrt(self.dt)
    drift = self.drift
    r = np.repeat(np.repeat(np.repeat(
        np.reshape(drift, (-1, 1, 1)), nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    sig = np.repeat(np.repeat(np.repeat(
        np.reshape(self.volatility, (-1, 1, 1)), nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    spot_paths[:, :,  1:] = np.repeat(
        spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(np.cumsum(
        r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2))
    # dimensions: [nb_paths, nb_stocks, nb_dates+1]
    if return_dW:
        return spot_paths, None, dW
    return spot_paths, None

  def generate_paths_with_alternatives(
          self, nb_paths=None, nb_alternatives=1, nb_dates=None):
    """Returns a nparray (nb_paths * nb_stocks * nb_dates) with prices."""
    nb_paths = nb_paths or self.nb_paths
    nb_dates = nb_dates or self.nb_dates
    total_nb_paths = nb_paths + nb_paths * nb_alternatives * nb_dates
    spot_paths = np.empty((total_nb_paths, self.nb_stocks, nb_dates+1))
    spot_paths[:, :, 0] = self.spot
    random_numbers = np.random.normal(
        0, 1, (total_nb_paths, self.nb_stocks, nb_dates))
    mult = nb_alternatives * nb_paths
    for i in range(nb_dates-1):
        random_numbers[
            nb_paths+i*mult:nb_paths+(i+1)*mult,:,:nb_dates-i-1] = np.tile(
            random_numbers[:nb_paths, :, :nb_dates-i-1],
            reps=(nb_alternatives, 1, 1))
    dW = random_numbers * np.sqrt(self.dt)
    drift = self.drift
    r = np.repeat(np.repeat(np.repeat(
        np.reshape(drift, (-1, 1, 1)), total_nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    sig = np.repeat(np.repeat(np.repeat(
        np.reshape(self.volatility, (-1, 1, 1)), total_nb_paths, axis=0),
        self.nb_stocks, axis=1), nb_dates, axis=2)
    spot_paths[:, :,  1:] = np.repeat(
        spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(np.cumsum(
        r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2))
    # dimensions: [nb_paths, nb_stocks, nb_dates+1]
    return spot_paths, None



####################################
# Heston
#####################################

class Heston(Model):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model
    Diffusion of the stock: dS = mu*S*dt + sqrt(v)*S*dW
    Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
    """
    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot,  maturity=maturity, dividend=dividend, name="Heston"
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def drift_fct(self, x, t):
      del t
      return self.drift * x

    def diffusion_fct(self, x, t, v=0):
      del t
      v_positive = np.maximum(v, 0)
      return np.sqrt(v_positive) * x

    def var_drift_fct(self, x, v):
      v_positive = np.maximum(v, 0)
      return - self.speed * (np.subtract(v_positive,self.mean))

    def var_diffusion_fct(self, x, v):
      v_positive = np.maximum(v, 0)
      return self.volatility * np.sqrt(v_positive)

    def generate_paths(self, start_X=None):
        paths = np.empty(
            (self.nb_paths, self.nb_stocks, self.nb_dates + 1))
        var_paths = np.empty(
            (self.nb_paths, self.nb_stocks, self.nb_dates + 1))

        dt = self.maturity / self.nb_dates
        if start_X is not None:
          paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
          if start_X is None:
            paths[i, :, 0] = self.spot
            var_paths[i, :, 0] = self.mean
            for k in range(1, self.nb_dates + 1):
                normal_numbers_1 = np.random.normal(0, 1, self.nb_stocks)
                normal_numbers_2 = np.random.normal(0, 1, self.nb_stocks)
                dW = normal_numbers_1 * np.sqrt(dt)
                dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                    1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                var_paths[i, :, k] = (
                        var_paths[i, :, k - 1]
                        + self.var_drift_fct(paths[i, :, k - 1],
                                             var_paths[i, :, k - 1], ) * dt
                        + np.multiply(
                    self.var_diffusion_fct(paths[i, :, k - 1],
                                           var_paths[i, :, k - 1]), dZ))

                paths[i, :, k] = (
                        paths[i, :, k - 1]
                        + self.drift_fct(paths[i, :, k - 1],
                                        (k-1) * dt) * dt
                        + np.multiply(self.diffusion_fct(paths[i, :, k - 1],
                                                    (k) * dt,
                                                    var_paths[i, :, k]), dW))
        return paths, var_paths


    def draw_path_heston(self, filename):
        nb_paths = self.nb_paths
        self.nb_paths = 1
        paths = self.generate_paths()
        self.nb_paths = nb_paths
        paths, var_paths = paths
        one_path = paths[0, 0, :]
        one_var_path = var_paths[0, 0, :]
        dates = np.array([i for i in range(len(one_path))])
        dt = self.maturity / self.nb_dates
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('Stock', color=color)
        ax1.plot(dates, one_path, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Variance', color=color)
        ax2.plot(dates, one_var_path, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()


class HestonWithVar(Heston):
    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(HestonWithVar, self).__init__(
            drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
            nb_dates, spot, maturity, dividend=dividend, sine_coeff=sine_coeff,
            **kwargs
        )
        self.return_var = True
        self.var_dim = self.nb_stocks


    
#############################
# Electricity markets
#############################
import numpy as np
import math

class ElectricityMarketModel(Model):
    """
    Electricity spot model:
        S_t = exp( f(t) + X_t + Y_t )
        dX_t = -alpha X_t dt + sigma dW_t
        dY_t = -beta Y_{t-} dt + J dN_t,  N Poisson intensity lam

    Conventions:
      - stock_paths: (nb_paths, nb_stocks, nb_dates+1)
      - var_paths:   (nb_paths, 2*nb_stocks, nb_dates+1) containing [X, Y] factors
      - return_var = True so pricer can append var_paths as extra features.
    """

    def __init__(
        self,
        rate: float,          # risk-free rate used for discounting in pricer
        alpha: float,
        sigma: float,
        beta: float,
        lam: float,
        x0: float,
        y0: float,
        nb_paths: int,
        nb_stocks: int,
        nb_dates: int,
        maturity: float,
        dist_par: float,
        # deterministic f(t) parameters (default matches your example)
        f_level: float = 100.0,   # used as log(f_level)
        f_amp: float = 0.5,
        f_period: float = 1.0,    # period in same units as t (e.g. 1 year)
        dividend: float = 0.0,
        **kwargs,
    ):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.beta  = float(beta)
        self.lam   = float(lam)
        self.x0    = float(x0)
        self.y0    = float(y0)
        self.dist_par = float(dist_par)

        self.f_level  = float(f_level)
        self.f_amp    = float(f_amp)
        self.f_period = float(f_period)

        # Compute S0 implied by f(0), x0, y0 (so Model.spot is consistent)
        f0 = self._f_det(0.0)
        s0 = float(np.exp(f0 + self.x0 + self.y0))

        super(ElectricityMarketModel,self).__init__(
            drift=rate, dividend=dividend, volatility=0.0,  # volatility not used here
            spot=s0,
            nb_stocks=nb_stocks,
            nb_paths=nb_paths,
            nb_dates=nb_dates,
            maturity=maturity,
            name="ElectricityMarket",
        )

        # Tell the pricer we have extra state variables in var_paths
        self.return_var = True
        self.var_dim = 2 * self.nb_stocks

    # ---- deterministic seasonality ----
    def _f_det(self, t):
        # Your example: log(100) + 0.5*cos(2*pi*t)
        # Here we allow a period: cos(2*pi*t/period)
        return math.log(self.f_level) + self.f_amp * math.cos(2.0 * math.pi * t / self.f_period)

    def _f_grid(self, nb_dates=None):
        nb_dates = nb_dates or self.nb_dates
        t = np.linspace(0.0, self.maturity, nb_dates + 1)
        # vectorize _f_det
        return np.log(self.f_level) + self.f_amp * np.cos(2.0 * np.pi * t / self.f_period)

    # ---- OU drift/diffusion (not required by pricer, but keeps interface consistent) ----
    def drift_fct(self, x, t):
        return -self.alpha * x

    def diffusion_fct(self, x, t, v=0):
        return self.sigma

    # ---- jump size sampler: placeholder for now ----
    def _sample_jump_sizes(self, n: int):
        """
        Placeholder. Replace later with your own distributions.
        For now: exponential(mean=1.0).
        """
        return np.random.exponential(scale=self.dist_par, size=n)

    def generate_paths(self, nb_paths=None, return_dW=False, dW=None, nb_dates=None):
        nb_paths = nb_paths or self.nb_paths
        nb_dates = nb_dates or self.nb_dates
        dt = self.maturity / nb_dates

        # factors
        X = np.empty((nb_paths, self.nb_stocks, nb_dates + 1), dtype=float)
        Y = np.empty((nb_paths, self.nb_stocks, nb_dates + 1), dtype=float)
        X[:, :, 0] = self.x0
        Y[:, :, 0] = self.y0

        # --- Brownian increments for X (optional external dW like BlackScholes) ---
        if dW is None:
            Z = np.random.normal(0.0, 1.0, size=(nb_paths, self.nb_stocks, nb_dates))
            dW_used = Z * math.sqrt(dt)
        else:
            dW_arr = np.asarray(dW, dtype=float)
            # accept either (nb_paths, nb_stocks, nb_dates) or (nb_paths, nb_dates) if nb_stocks==1
            if dW_arr.ndim == 2 and self.nb_stocks == 1:
                dW_arr = dW_arr[:, None, :]
            if dW_arr.shape != (nb_paths, self.nb_stocks, nb_dates):
                raise ValueError("dW must have shape (nb_paths, nb_stocks, nb_dates).")
            dW_used = dW_arr
            Z = dW_used / math.sqrt(dt)

        # Exact OU transition constants for X
        aX = math.exp(-self.alpha * dt)
        qX = (1.0 - math.exp(-2.0 * self.alpha * dt)) / (2.0 * self.alpha) if self.alpha > 0 else dt
        stdX = self.sigma * math.sqrt(qX)

        # Jump decay constant for Y
        aY = math.exp(-self.beta * dt)

        for k in range(nb_dates):
            # X exact step
            X[:, :, k + 1] = aX * X[:, :, k] + stdX * Z[:, :, k]

            # Y step: decay + Poisson jumps inside interval
            y_next = aY * Y[:, :, k]
            Nk = np.random.poisson(self.lam * dt, size=(nb_paths, self.nb_stocks))

            # add jump contributions where Nk > 0
            idx = np.argwhere(Nk > 0)
            for (i, j) in idx:
                n = int(Nk[i, j])
                U = np.random.uniform(0.0, dt, size=n)  # jump times in the step
                J = np.asarray(self._sample_jump_sizes(n), dtype=float)
                y_next[i, j] += np.sum(np.exp(-self.beta * (dt - U)) * J)

            Y[:, :, k + 1] = y_next

        # Build S = exp(f + X + Y)
        fgrid = self._f_grid(nb_dates=nb_dates)  # (nb_dates+1,)
        logS = fgrid[None, None, :] + X + Y
        S = np.exp(logS)

        # var_paths: pack X and Y along "feature axis" -> (nb_paths, 2*nb_stocks, nb_dates+1)
        var_paths = np.concatenate([X, Y], axis=1)

        if return_dW:
            return S, var_paths, dW_used
        return S, var_paths


# ==============================================================================
# dict for the supported stock models to get them from their name
STOCK_MODELS = {
    "BlackScholes": BlackScholes,
    "Heston": Heston,
    "HestonWithVar": HestonWithVar,
    "ElectricityMarketModel": ElectricityMarketModel,
}
# ==============================================================================


#hyperparam_test_stock_models = {
#    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5, 'speed': 0.5, 'hurst':0.05,
#    'correlation': 0.5, 'nb_paths': 1, 'nb_dates': 100, 'maturity': 1.,
#    'nb_stocks':10, 'spot':100}
#
#def draw_stock_model(stock_model_name):
#    hyperparam_test_stock_models['model_name'] = stock_model_name
#    stockmodel = STOCK_MODELS[stock_model_name](**hyperparam_test_stock_models)
#    stock_paths = stockmodel.generate_paths()
#    filename = '{}.pdf'.format(stock_model_name)
#
#    # draw a path
#    one_path = stock_paths[0, 0, :]
#    dates = np.array([i for i in range(len(one_path))])
#    plt.plot(dates, one_path, label='stock path')
#    plt.legend()
#    plt.savefig(filename)
#    plt.close()
#
#if __name__ == '__main__':
#    # draw_stock_model("BlackScholes")
#    # draw_stock_model("FractionalBlackScholes")
#    # heston = STOCK_MODELS["Heston"](**hyperparam_test_stock_models)
#    # heston.draw_path_heston("heston.pdf")
#
#    rHeston = RoughHeston(**hyperparam_test_stock_models)
#    t = time.time()
#    p = rHeston.generate_paths(1000)
#    print("needed time: {}".format(time.time()-t))
#

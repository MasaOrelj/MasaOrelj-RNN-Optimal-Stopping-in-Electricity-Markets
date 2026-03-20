# Lint as: python3
"""
Main module to run the algorithms.
"""
import os

import atexit
import csv
import itertools
import multiprocessing
import socket
import random
import time
import psutil

# absl needs to be upgraded to >= 0.10.0, otherwise joblib might not work
from absl import app
from absl import flags
import numpy as np
import shutil

from optimal_stopping.utilities import configs_getter
from optimal_stopping.algorithms.backward_induction import DOS
from optimal_stopping.payoffs import payoff
from optimal_stopping.algorithms.backward_induction import LSM
from optimal_stopping.algorithms.backward_induction import RLSM
from optimal_stopping.algorithms.backward_induction import RRLSM
from optimal_stopping.data import stock_model
from optimal_stopping.algorithms.backward_induction import NLSM
from optimal_stopping.algorithms.reinforcement_learning import RFQI
from optimal_stopping.algorithms.reinforcement_learning import FQI
from optimal_stopping.algorithms.reinforcement_learning import LSPI
from optimal_stopping.algorithms.finite_difference import binomial
from optimal_stopping.algorithms.finite_difference import trinomial
from optimal_stopping.run import write_figures
from optimal_stopping.algorithms.backward_induction import backward_induction_pricer
from optimal_stopping.run import configs
from optimal_stopping.algorithms.backward_induction import LSM_swing_implementation


import joblib

# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = SendBotMessage()

NUM_PROCESSORS = multiprocessing.cpu_count()
if 'ada-' in socket.gethostname() or 'arago' in socket.gethostname():
    SERVER = True
    NB_JOBS = int(NUM_PROCESSORS) - 1
else:
    SERVER = False
    NB_JOBS = int(NUM_PROCESSORS) - 1


SEND = False
if SERVER:
    SEND = True





FLAGS = flags.FLAGS

flags.DEFINE_list("nb_stocks", None, "List of number of Stocks")
flags.DEFINE_list("algos", None, "Name of the algos to run.")
flags.DEFINE_bool("print_errors", False, "Set to True to print errors if any.")
flags.DEFINE_integer("nb_jobs", NB_JOBS, "Number of CPUs to use parallelly")
flags.DEFINE_bool("generate_pdf", False, "Whether to generate latex tables")
flags.DEFINE_integer("path_gen_seed", None,
                     "Seed for path generation")
flags.DEFINE_bool("compute_upper_bound", False,
                  "Whether to additionally compute upper bound for price")
flags.DEFINE_bool("compute_greeks", False,
                  "Whether to compute greeks (not available for all settings)")
flags.DEFINE_string("greeks_method", "central",
                    "one of: central, forward, backward, regression")
flags.DEFINE_float("eps", 1e-9,
                   "the epsilon for the finite difference method or regression")
flags.DEFINE_float("reg_eps", 5,
                   "the epsilon for the finite difference method or regression")
flags.DEFINE_integer("poly_deg", 2,
                     "the degree for the polynomial regression")
flags.DEFINE_bool("fd_freeze_exe_boundary", True,
                  "Whether to use same exercise boundary")
flags.DEFINE_bool("fd_compute_gamma_via_PDE", True,
                  "Whether to use the PDE to compute gamma")
flags.DEFINE_bool("DEBUG", False, "Turn on debug mode")
flags.DEFINE_integer("train_eval_split", 2,
                     "divisor for the train/eval split")


_CSV_HEADERS = ['algo', 'model', 'payoff', 'drift', 'volatility', 'mean',
                'speed', 'correlation', 'hurst', 'nb_stocks',
                'nb_paths', 'nb_dates', 'spot', 'strike', 'dividend',
                'maturity', 'nb_epochs', 'hidden_size', 'factors',
                'ridge_coeff', 'use_payoff_as_input',
                'train_ITM_only', 'use_path', 'use_var',
                'price', 'duration', 'time_path_gen', 'comp_time',
                'delta', 'gamma', 'theta', 'rho', 'vega', 'greeks_method',
                # electricity model parameters
                'rate', 'alpha', 'sigma', 'beta', 'lam', 'x0', 'y0',
                'dist_par', 'f_level', 'f_amp', 'f_period',
                'num_swings', 'exercise_dates',
                'price_upper_bound',]

_PAYOFFS = {
    "MaxPut": payoff.MaxPut,
    "MaxCall": payoff.MaxCall,
    "GeometricPut": payoff.GeometricPut,
    "BasketCall": payoff.BasketCall,
    "Identity": payoff.Identity,
    "Max": payoff.Max,
    "Mean": payoff.Mean,
    "MinPut": payoff.MinPut,
    "Put1Dim": payoff.Put1Dim,
    "Call1Dim": payoff.Call1Dim,
}

_STOCK_MODELS = stock_model.STOCK_MODELS

_ALGOS = {
    "LSM": LSM.LeastSquaresPricer,
    "FQI": FQI.FQIFast,
    "NLSM": NLSM.NeuralNetworkPricer,
    #"DOS": DOS.DeepOptimalStopping,
    "RLSM": RLSM.ReservoirLeastSquarePricerFast,
    "SwingLSM": LSM_swing_implementation.SwingLeastSquaresPricer,
    "SwingRLSM": LSM_swing_implementation.SwingReservoirLeastSquarePricerFast,
    #"RLSMTanh": RLSM.ReservoirLeastSquarePricerFastTanh,
    #"RLSMRidge": RLSM.ReservoirLeastSquarePricerFastRidge,
    #"RLSMElu": RLSM.ReservoirLeastSquarePricerFastELU,
    #"RLSMSilu": RLSM.ReservoirLeastSquarePricerFastSILU,
    #"RLSMGelu": RLSM.ReservoirLeastSquarePricerFastGELU,
    #"RLSMSoftplus": RLSM.ReservoirLeastSquarePricerFastSoftplus,
    #"RLSMSoftplusReinit": RLSM.ReservoirLeastSquarePricerFastSoftplusReinit,

    #"RRLSM": RRLSM.ReservoirRNNLeastSquarePricer2,
    #"RRLSMmix": RRLSM.ReservoirRNNLeastSquarePricer,
    #"RRLSMRidge": RRLSM.ReservoirRNNLeastSquarePricer2Ridge,

    "RFQI": RFQI.FQI_ReservoirFast,
    #"RFQISoftplus": RFQI.FQI_ReservoirFastSoftplus,
    #"RFQITanh": RFQI.FQI_ReservoirFastTanh,
    #"RFQIRidge": RFQI.FQI_ReservoirFastRidge,
    #"RFQILasso": RFQI.FQI_ReservoirFastLasso,

    #"RRFQI": RFQI.FQI_ReservoirFastRNN,

    #"EOP": backward_induction_pricer.EuropeanOptionPricer,
    #"B": binomial.BinomialPricer,
    #"Trinomial": trinomial.TrinomialPricer,
}

_NUM_FACTORS = {
    "RRLSMmix": 3,
    "RRLSM": 2,
    "RLSM": 1,
    "SwingRLSM": 1,
}



def init_seed():
  random.seed(0)
  np.random.seed(0)


def _cfg_list(cfg, name):
    """Return an iterable for `name` on cfg; fallback to (None,) if missing."""
    return getattr(cfg, name) if hasattr(cfg, name) else (None,)


def _run_algos():
  fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft",
                       f'{int(time.time()*1000)}.csv')
  tmp_dirpath = f'{fpath}.tmp_results'
  os.makedirs(tmp_dirpath, exist_ok=True)
  atexit.register(lambda: shutil.rmtree(tmp_dirpath, ignore_errors=True))
  #atexit.register(shutil.rmtree, tmp_dirpath)
  tmp_files_idx = 0

  delayed_jobs = []

  nb_stocks_flag = [int(nb) for nb in FLAGS.nb_stocks or []]
  for config_name, config in configs_getter.get_configs():
    print(f'Config {config_name}', config)
    config.algos = [a for a in config.algos
                    if FLAGS.algos is None or a in FLAGS.algos]
    if nb_stocks_flag:
      config.nb_stocks = [a for a in config.nb_stocks
                          if a in nb_stocks_flag]
    combinations = list(itertools.product(
        config.algos, config.dividends, config.maturities, config.nb_dates,
        config.nb_paths, config.nb_stocks, config.payoffs, config.drift,
        config.spots, config.stock_models, config.strikes, config.volatilities,
        config.mean, config.speed, config.correlation, config.hurst,
        # optional electricity parameters (use (None,) when not present)
        _cfg_list(config, 'rate'), _cfg_list(config, 'alpha'), _cfg_list(config, 'sigma'),
        _cfg_list(config, 'beta'), _cfg_list(config, 'lam'), _cfg_list(config, 'x0'),
        _cfg_list(config, 'y0'), _cfg_list(config, 'dist_par'), _cfg_list(config, 'f_level'),
        _cfg_list(config, 'f_amp'), _cfg_list(config, 'f_period'),
        _cfg_list(config, 'num_swings'),
        _cfg_list(config, 'exercise_dates'),
        config.nb_epochs, config.hidden_size, config.factors,
        config.ridge_coeff,
        config.train_ITM_only, config.use_path, config.use_payoff_as_input,
        _cfg_list(config, 'use_var'),
        _cfg_list(config, 'use_spot_as_input')))

    # random.shuffle(combinations)
    for params in combinations:
      for i in range(config.nb_runs):
        tmp_file_path = os.path.join(tmp_dirpath, str(tmp_files_idx))
        tmp_files_idx += 1
        delayed_jobs.append(joblib.delayed(_run_algo)(
            tmp_file_path, *params, fail_on_error=FLAGS.print_errors,
            compute_greeks=FLAGS.compute_greeks,
            greeks_method=FLAGS.greeks_method,
            eps=FLAGS.eps, poly_deg=FLAGS.poly_deg,
            fd_freeze_exe_boundary=FLAGS.fd_freeze_exe_boundary,
            fd_compute_gamma_via_PDE=FLAGS.fd_compute_gamma_via_PDE,
            reg_eps=FLAGS.reg_eps, path_gen_seed=FLAGS.path_gen_seed,
            compute_upper_bound=FLAGS.compute_upper_bound,
            DEBUG=FLAGS.DEBUG, train_eval_split=FLAGS.train_eval_split,))

  print(f"Running {len(delayed_jobs)} tasks using "
        f"{FLAGS.nb_jobs}/{NUM_PROCESSORS} CPUs...")
  joblib.Parallel(n_jobs=FLAGS.nb_jobs)(delayed_jobs)

  print(f'Writing results to {fpath}...')
  with open(fpath, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=_CSV_HEADERS)
    writer.writeheader()
    for idx in range(tmp_files_idx):
      tmp_file_path = os.path.join(tmp_dirpath, str(idx))
      try:
        with open(tmp_file_path,  "r") as read_f:
          csvfile.write(read_f.read())
      except FileNotFoundError:
        pass

    # Aggregate per-run stopping-time files (if present) into one CSV
    stops_fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft",
                                                         f'{int(time.time()*1000)}_stopping_times.csv')
    aggregated_rows = []
    counts_headers = None

    # helper to parse a stops CSV and extract algorithm,num_dim and counts
    def _parse_stops_lines(lines):
        parts = [l.strip() for l in lines if l.strip()]
        if len(parts) < 2:
            return None
        header_parts = parts[0].split(',')
        row_parts = parts[1].split(',')
        # Expect header: algorithm,num_dim,model_name,strike,spot,1,2,...
        if len(header_parts) < 6 or len(row_parts) < 5:
            return None
        algo = row_parts[0]
        num_dim = row_parts[1]
        model_name = row_parts[2]
        strike = row_parts[3]
        spot = row_parts[4]
        counts = row_parts[5:]
        return algo, num_dim, model_name, strike, spot, counts, header_parts[5:]

    for idx in range(tmp_files_idx):
        tmp_stops = os.path.join(tmp_dirpath, str(idx) + '.stops.csv')
        try:
            with open(tmp_stops, 'r') as sf:
                parsed = _parse_stops_lines(sf.readlines())
                if parsed is None:
                    continue
                algo, num_dim, model_name, strike, spot, counts, hdr_counts = parsed
                if counts_headers is None:
                    counts_headers = hdr_counts
                aggregated_rows.append((algo, num_dim, model_name, strike, spot, counts))
        except FileNotFoundError:
            continue

    # also include any stops files directly written into output/metrics_draft
    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../output/metrics_draft"))
    try:
        for fname in os.listdir(out_dir):
            if fname.endswith('_stops.csv'):
                fpath = os.path.join(out_dir, fname)
                try:
                    with open(fpath, 'r') as sf:
                        parsed = _parse_stops_lines(sf.readlines())
                        if parsed is None:
                            continue
                        algo, num_dim, model_name, strike, spot, counts, hdr_counts = parsed
                        if counts_headers is None:
                            counts_headers = hdr_counts
                        # avoid duplicates
                        if (algo, num_dim, model_name, strike, spot, counts) not in aggregated_rows:
                            aggregated_rows.append((algo, num_dim, model_name, strike, spot, counts))
                except Exception:
                    continue
    except Exception:
        pass

    if counts_headers is not None and aggregated_rows:
        stops_fpath = os.path.normpath(stops_fpath)
        with open(stops_fpath, 'w') as out:
            header = ['algorithm', 'num_dim', 'model_name', 'strike', 'spot'] + counts_headers
            out.write(','.join(header) + '\n')
            for algo, num_dim, model_name, strike, spot, counts in aggregated_rows:
                # pad or trim counts to match header length
                wanted = len(counts_headers)
                counts = counts[:wanted] + ['0']*(max(0, wanted - len(counts)))
                row = [str(algo), str(num_dim), str(model_name), str(strike), str(spot)] + counts
                out.write(','.join(row) + '\n')

  return fpath


def _run_algo(
    metrics_fpath, algo, dividend, maturity, nb_dates, nb_paths,
    nb_stocks, payoff, drift, spot, stock_model, strike, volatility, mean,
    speed, correlation, hurst,
    # optional electricity params
    rate=None, alpha=None, sigma=None, beta=None, lam=None,
    x0=None, y0=None, dist_par=None, f_level=None, f_amp=None,
    f_period=None, num_swings=None, exercise_dates=None,
    nb_epochs=None, hidden_size=10,
    factors=(1.,1.,1.), ridge_coeff=1.,
    train_ITM_only=True, use_path=False, use_payoff_as_input=False,
    use_var=None, use_spot_as_input=True,
    fail_on_error=False,
    compute_greeks=False, greeks_method=None, eps=None,
    poly_deg=None, fd_freeze_exe_boundary=True,
    fd_compute_gamma_via_PDE=True, reg_eps=None, path_gen_seed=None,
    compute_upper_bound=False,
    DEBUG=False, train_eval_split=2):

  """
  This functions runs one algo for option pricing. It is called by _run_algos()
  which is called in main(). Below the inputs are listed which have to be
  specified in the config that is passed to main().

  Args:
   metrics_fpath: file path, automatically generated & passed by
            _run_algos()
                fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft",
                                                         f'{int(time.time()*1000)}.csv')
                tmp_dirpath = f'{fpath}.tmp_results'
                os.makedirs(tmp_dirpath, exist_ok=True)
                def _cleanup_tmp_dir(path):
                    try:
                        shutil.rmtree(path)
                    except Exception:
                        try:
                            # try again but ignore errors (best-effort)
                            shutil.rmtree(path, ignore_errors=True)
                        except Exception:
                            pass

                atexit.register(_cleanup_tmp_dir, tmp_dirpath)
   mean (float): parameter for Heston stock model.
   speed (float): parameter for Heston stock model.
   correlation (float): parameter for Heston stock model.
   hurst (float): in (0,1) the hurst parameter for the
            FractionalBrownianMotion and FractionalBrownianMotionPathDep stock
            models.
   nb_epochs (int): number of epochs to train the algos which are based on
            iterative updates (FQI, RFQI etc.) or SGD (NLSM, DOS). Otherwise
            unused.
   hidden_size (int): the number of nodes in the hidden layers of NN based
            algos.
   factors (list of floats, optional. Contains scaling coeffs for the
            randomized NN (i.e. scaling of the randomly sampled and fixed
            weights -> changes the std of the sampling distribution). Depending
            on the algo, the factors are used differently. See there directly.
   ridge_coeff (float, regression coeff for the algos using Ridge
            regression.
   train_ITM_only (bool): whether to train weights on all paths or only on
            those where payoff is positive (i.e. where it makes sense to stop).
            This should be set to False when using FractionalBrownianMotion with
            Identity payoff, since there the payoff can actually become negative
            and therefore training on those paths is important.
   use_path (bool): for DOS algo only. If true, the algo uses the entire
            path up to the current time of the stock (instead of the current
            value only) as input. This is used for Non-Markovian stock models,
            i.e. FractionalBrownianMotion, where the decisions depend on the
            history.
   fail_on_error (bool): whether to continue when errors occure or not.
            Automatically passed from _run_algos(), with the value of
            FLAGS.print_errors.
   compute_greeks (bool): whether to compute the greeks (delta and gamma),
            not available in every setting
   greeks_method (string): method for finite difference methods to compute greeks,
            one of {'central', 'forward', 'backward'}
   eps (float): the epsilon to use in finite difference method to compute
            greeks
   poly_deg (int): the degree for the polynomial regression usid in regression
            based greeks computation
   fd_freeze_exe_boundary (bool): whether to use same exersice boundary or not
   fd_compute_gamma_via_PDE (bool): whether to compute gamma via the PDE
   reg_eps (float): the epsilon to use in the regression method to compute
            greeks
   path_gen_seed (int or None): seed for path generation (if not None)
   compute_upper_bound (bool): whether to compute the upper bound for the price
  """
  if path_gen_seed is not None:
    configs.path_gen_seed.set_seed(path_gen_seed)
  print(algo, spot, volatility, maturity, nb_paths, '... ', end="")
  payoff_ = _PAYOFFS[payoff](strike)
  # assemble kwargs so electricity model params are passed when available
  stock_kwargs = dict(
      drift=drift, volatility=volatility, mean=mean, speed=speed, hurst=hurst,
      correlation=correlation, nb_stocks=nb_stocks,
      nb_paths=nb_paths, nb_dates=nb_dates,
      spot=spot, dividend=dividend,
      maturity=maturity, name=stock_model,
      rate=rate, alpha=alpha, sigma=sigma, beta=beta, lam=lam,
      x0=x0, y0=y0, dist_par=dist_par, f_level=f_level, f_amp=f_amp,
      f_period=f_period,
  )
  stock_model_ = _STOCK_MODELS[stock_model](**stock_kwargs)
  if algo in ['NLSM']:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size,
                            train_ITM_only=train_ITM_only,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["LND", "LN", "LNfast", "LN2"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["DOS", "pathDOS"]:
      if algo == "DOS" and use_path:
          print("change use_path to 'False', otherwise use the algo 'pathDOS'")
          use_path = False
      if algo == "pathDOS" and not use_path:
          print("change use_path to 'True', otherwise use the algo 'DOS'")
          use_path = True
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size, use_path=use_path,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["RLSM", "RRLSMmix", "RRLSM", "RLSMTanh", "RLSMElu", "RLSMSilu",
                "RLSMGelu","RLSMSoftplus", "RLSMSoftplusReinit",
                "RRFQI", "RFQITanh", "RFQI",]:
    pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                    hidden_size=hidden_size, factors=factors,
                    train_ITM_only=train_ITM_only,
                    use_payoff_as_input=use_payoff_as_input,
                    use_spot_as_input=use_spot_as_input,
                    use_var=use_var)
  elif algo in ["RLSMRidge", "RFQIRidge", "RRLSMRidge"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            hidden_size=hidden_size, factors=factors,
                            train_ITM_only=train_ITM_only,
                            ridge_coeff=ridge_coeff,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["FQIRidge"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            ridge_coeff=ridge_coeff,
                            train_ITM_only=train_ITM_only,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["LSM", "LSMDeg1", "LSMLaguerre"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            train_ITM_only=train_ITM_only,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["LSMRidge"]:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            train_ITM_only=train_ITM_only,
                            ridge_coeff=ridge_coeff,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif "FQI" in algo:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            train_ITM_only=train_ITM_only,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)
  elif algo in ["SwingLSM"]:
    pricer = _ALGOS[algo](
        stock_model_, payoff_,
        nb_epochs=nb_epochs,
        train_ITM_only=train_ITM_only,
        use_payoff_as_input=use_payoff_as_input,
        use_spot_as_input=use_spot_as_input,
        use_var=use_var,
        num_swings=num_swings,
        exercise_dates=exercise_dates,
    )

  elif algo in ["SwingRLSM"]:
      pricer = _ALGOS[algo](
          stock_model_, payoff_,
          hidden_size=hidden_size,
          factors=factors,
          nb_epochs=nb_epochs,
          train_ITM_only=train_ITM_only,
          use_payoff_as_input=use_payoff_as_input,
          use_spot_as_input=use_spot_as_input,
          use_var=use_var,
          num_swings=num_swings,
          exercise_dates=exercise_dates,
      )
  else:
      pricer = _ALGOS[algo](stock_model_, payoff_, nb_epochs=nb_epochs,
                            use_payoff_as_input=use_payoff_as_input,
                            use_spot_as_input=use_spot_as_input,
                            use_var=use_var)

    # Ensure pricer instances get the `use_spot_as_input` flag when their
    # constructors don't accept it. Recompute input_dim accordingly.
  try:
          from optimal_stopping.algorithms.backward_induction import \
                  backward_induction_pricer as _bip
          # set attributes so algorithms can consult them
          setattr(pricer, 'use_spot_as_input', use_spot_as_input)
          setattr(pricer, 'use_var', use_var)
          # recompute input_dim to reflect the chosen setting
          setattr(pricer, 'input_dim', _bip.compute_input_dim(
                  getattr(pricer, 'model', stock_model_),
                  getattr(pricer, 'use_var', False),
                  getattr(pricer, 'use_payoff_as_input', False),
                  use_spot_as_input))
  except Exception:
            pass

  # expose a tmp-file prefix and some metadata so pricers can write
  # companion files (e.g. stopping-time histograms) into the same
  # tmp-results folder used by the aggregator
  try:
      setattr(pricer, '_metrics_tmp_prefix', metrics_fpath)
      setattr(pricer, '_algo_name', algo)
      setattr(pricer, '_nb_stocks_val', nb_stocks)
      setattr(pricer, '_model_name_val', stock_model)
  except Exception:
      pass

  t_begin = time.time()

  if DEBUG:
      if compute_upper_bound:
          price, price_u, gen_time = pricer.price_upper_lower_bound(
              train_eval_split=train_eval_split)
          delta, gamma, theta, rho, vega = [None] * 5
      elif not compute_greeks:
          price, gen_time = pricer.price(train_eval_split=train_eval_split)
          delta, gamma, theta, rho, vega, price_u = [None] * 6
      else:
          price, gen_time, delta, gamma, theta, rho, vega = pricer.price_and_greeks(
              eps=eps, greeks_method=greeks_method, poly_deg=poly_deg,
              reg_eps=reg_eps,
              fd_freeze_exe_boundary=fd_freeze_exe_boundary,
              fd_compute_gamma_via_PDE=fd_compute_gamma_via_PDE,
              train_eval_split=train_eval_split)
          price_u = None
      duration = time.time() - t_begin
      comp_time = duration - gen_time
      return

  try:
    if compute_upper_bound:
        price, price_u, gen_time = pricer.price_upper_lower_bound(
            train_eval_split=train_eval_split)
        delta, gamma, theta, rho, vega = [None] * 5
    elif not compute_greeks:
        price, gen_time = pricer.price(train_eval_split=train_eval_split)
        delta, gamma, theta, rho, vega, price_u = [None]*6
    else:
        price, gen_time, delta, gamma, theta, rho, vega = pricer.price_and_greeks(
            eps=eps, greeks_method=greeks_method, poly_deg=poly_deg,
            reg_eps=reg_eps,
            fd_freeze_exe_boundary=fd_freeze_exe_boundary,
            fd_compute_gamma_via_PDE=fd_compute_gamma_via_PDE,
            train_eval_split=train_eval_split)
        price_u = None
    duration = time.time() - t_begin
    comp_time = duration - gen_time
  except BaseException as err:
    if fail_on_error:
      raise
    print(err)
    return
  metrics_ = {}
  metrics_['algo'] = algo
  metrics_['model'] = stock_model
  metrics_['payoff'] = payoff
  metrics_['drift'] = drift
  metrics_['volatility'] = volatility
  metrics_['mean'] = mean
  metrics_['speed'] = speed
  metrics_['correlation'] = correlation
  metrics_['hurst'] = hurst
  metrics_['nb_stocks'] = nb_stocks
  metrics_['nb_paths'] = nb_paths
  metrics_['nb_dates'] = nb_dates
  metrics_['spot'] = spot
  metrics_['strike'] = strike
  metrics_['dividend'] = dividend
  metrics_['maturity'] = maturity
  # electricity parameters
  metrics_['rate'] = rate
  metrics_['alpha'] = alpha
  metrics_['sigma'] = sigma
  metrics_['beta'] = beta
  metrics_['lam'] = lam
  metrics_['x0'] = x0
  metrics_['y0'] = y0
  metrics_['dist_par'] = dist_par
  metrics_['f_level'] = f_level
  metrics_['f_amp'] = f_amp
  metrics_['f_period'] = f_period
  metrics_['num_swings'] = num_swings
  metrics_['exercise_dates'] = exercise_dates
  metrics_['use_var'] = use_var
  metrics_['price'] = price
  metrics_['duration'] = duration
  metrics_['time_path_gen'] = gen_time
  metrics_['comp_time'] = comp_time
  metrics_['hidden_size'] = hidden_size
  metrics_['factors'] = factors
  metrics_['ridge_coeff'] = ridge_coeff
  metrics_['nb_epochs'] = nb_epochs
  metrics_['train_ITM_only'] = train_ITM_only
  metrics_['use_path'] = use_path
  metrics_['use_payoff_as_input'] = use_payoff_as_input
  metrics_['delta'] = delta
  metrics_['gamma'] = gamma
  metrics_['theta'] = theta
  metrics_['rho'] = rho
  metrics_['vega'] = vega
  metrics_['greeks_method'] = greeks_method
  metrics_['price_upper_bound'] = price_u
  print("price: ", price, "price upper: ", price_u,
        "computation-time: ", comp_time,
        "delta: ", delta, "gamma: ", gamma, "theta: ", theta, "rho: ", rho,
        "vega: ", vega)
  with open(metrics_fpath, "w") as metrics_f:
    writer = csv.DictWriter(metrics_f, fieldnames=_CSV_HEADERS)
    writer.writerow(metrics_)
    # write per-run stopping times histogram to a companion file so the
    # aggregator can combine them into a single CSV for the experiment
    # write per-run stopping times histogram to a companion file so the
# aggregator can combine them into a single CSV for the experiment
    try:
        all_ex = getattr(pricer, 'all_exercise_dates', None)

        if all_ex is not None:
            # swing case: count ALL exercises by date on evaluation paths
            split = getattr(pricer, 'split', None)
            if split is not None and split > 0 and split < len(all_ex):
                all_ex_to_count = all_ex[split:, :]
            else:
                all_ex_to_count = all_ex

            # ignore date 0, keep dates 1..nb_dates
            counts = np.sum(all_ex_to_count[:, 1:], axis=0).astype(int).tolist()

            stops_fpath = metrics_fpath + '.stops.csv'
            with open(stops_fpath, 'w') as sf:
                header = [
                    'algorithm', 'num_dim', 'model_name', 'strike', 'spot'
                ] + [str(i + 1) for i in range(nb_dates)]
                sf.write(','.join(header) + '\n')
                row = [
                    str(algo), str(nb_stocks), str(stock_model), str(strike), str(spot)
                ] + [str(c) for c in counts]
                sf.write(','.join(row) + '\n')

        else:
            # old one-stop case: keep existing behavior unchanged
            ex = getattr(pricer, 'last_ex_dates', None)
            if ex is not None:
                split = getattr(pricer, 'split', None)
                if split is not None and split > 0 and split < len(ex):
                    ex_to_count = ex[split:]
                else:
                    ex_to_count = ex
                ex = ex_to_count

                if ex.size and ex.min() == 0:
                    ex = ex + 1

                counts = [int(np.sum(ex == i)) for i in range(1, nb_dates + 1)]
                stops_fpath = metrics_fpath + '.stops.csv'
                with open(stops_fpath, 'w') as sf:
                    header = [
                        'algorithm', 'num_dim', 'model_name', 'strike', 'spot'
                    ] + [str(i + 1) for i in range(nb_dates)]
                    sf.write(','.join(header) + '\n')
                    row = [
                        str(algo), str(nb_stocks), str(stock_model), str(strike), str(spot)
                    ] + [str(c) for c in counts]
                    sf.write(','.join(row) + '\n')
    except Exception as e:
        print("stopping-times CSV error:", e)


def main(argv):
  del argv

  if FLAGS.DEBUG:
      configs.path_gen_seed.set_seed(FLAGS.path_gen_seed)
      filepath = _run_algos()
      return

  try:
      if SEND:
          SBM.send_notification(
              text='start running AMC2 with config:\n{}'.format(FLAGS.configs),
              chat_id="-399803347"
          )
      configs.path_gen_seed.set_seed(FLAGS.path_gen_seed)
      filepath = _run_algos()

      if FLAGS.generate_pdf:
          write_figures.write_figures()
          write_figures.generate_pdf()

      if SEND:
          time.sleep(1)
          SBM.send_notification(
              text='finished',
              files=[filepath],
              chat_id="-399803347"
          )
  except Exception as e:
      if SEND:
          SBM.send_notification(
              text='ERROR\n{}'.format(e),
              chat_id="-399803347"
          )
      else:
          print('ERROR\n{}'.format(e))




if __name__ == "__main__":
  app.run(main)

from optimal_stopping.data import stock_model 

import numpy as np
import matplotlib.pyplot as plt


el_model = stock_model.ElectricityMarketModel(
    rate=0.02,
    alpha=7.0,
    sigma=1.4,
    beta=200.0,
    lam=4.0,
    x0=1.0,
    y0=0.0,
    dist_par=0.4,
    nb_paths=20000,
    nb_stocks=1,
    nb_dates=400,
    maturity=2.0,
    f_level=100.0,
)

S, var_paths = el_model.generate_paths()

# Extract X and Y from var_paths
X = var_paths[:, 0, :]   # shape (nb_paths, nb_dates+1)
Y = var_paths[:, 1, :]   # shape (nb_paths, nb_dates+1)

# Deterministic seasonality on grid
f = el_model._f_grid()      # shape (nb_dates+1,)

# Broadcast f across paths if you want same matrix shape as X,Y
F = np.tile(f, (el_model.nb_paths, 1))   # shape (nb_paths, nb_dates+1)

# Spot paths as 2D matrix
S_mat = S[:, 0, :]       # shape (nb_paths, nb_dates+1)

# Parameters
times = np.linspace(0.0, 2.0, 401)
alpha = el_model.alpha
lam = el_model.lam
beta = el_model.beta
sigma = el_model.sigma
mu_J = el_model.dist_par
x0 = el_model.x0


# 1) sample paths

# X and Y
plt.figure(figsize=(9, 5))
for i in range(1):
        plt.plot(times, X[i], linewidth=1.0, alpha=0.8, label="X (OU)")
        plt.plot(times,Y[i],linewidth=1.0, alpha=0.8, label="Y")
plt.title("OU and Y process: sample path")
plt.xlabel("time (years)")
plt.ylabel("x(t), y(t)")
plt.legend()
plt.tight_layout()
plt.show()

#exp(f) and S
plt.figure(figsize=(9, 5))
for i in range(1):
        plt.plot(times, np.exp(f), linewidth=1.0, alpha=0.8, label="exp(f)")
        plt.plot(times, S_mat[i], linewidth=1.0, alpha=0.8, label="S")
plt.title("S and wxp(f) sample path")
plt.xlabel("time (years)")
plt.ylabel("exp(f(t)), S(t)")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Mean X
x_mean = X.mean(axis=0)
mean = x0*np.exp(-alpha*times)
plt.figure(figsize=(9, 5))
plt.plot(times, x_mean, label="sample mean x(t)")
plt.plot(times, mean, linestyle="--", label="mean(t) (theory)")
plt.title("Mean of x(t) vs deterministic mean")
plt.xlabel("time (years)")
plt.ylabel("level")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Mean Y
y_mean = Y.mean(axis=0)
mean = (lam/beta)*mu_J*(1-np.exp(-beta*times))
plt.figure(figsize=(9, 5))
plt.plot(times, y_mean, label="sample mean y(t)")
plt.plot(times, mean, linestyle="--", label="mean(t) (theory)")
plt.title("Mean of y(t) vs deterministic mean")
plt.xlabel("time (years)")
plt.ylabel("level")
plt.legend()
plt.tight_layout()
plt.show()


# 3) Variance X
x_var_emp = X.var(axis=0, ddof=1)
x_var_theory = (sigma**2/(2*alpha))*(1-np.exp(-2*alpha*times))
plt.figure(figsize=(9, 5))
plt.plot(times, x_var_emp, label="empirical Var[x(t)]")
plt.plot(times, x_var_theory, linestyle="--", label="theoretical Var[x(t)]")
plt.title("OU variance check")
plt.xlabel("time (years)")
plt.ylabel("variance")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Variance Y
y_var_emp = Y.var(axis=0, ddof=1)
y_var_theory = (lam/(2*beta))*(2*mu_J**2)*(1-np.exp(-2*beta*times))
plt.figure(figsize=(9, 5))
plt.plot(times, y_var_emp, label="empirical Var[y(t)]")
plt.plot(times, y_var_theory, linestyle="--", label="theoretical Var[y(t)]")
plt.title("Y variance check")
plt.xlabel("time (years)")
plt.ylabel("variance")
plt.legend()
plt.tight_layout()
plt.show()


# Analysis of Y distribution for t=1
t_star = 1.0
k = np.where(times == t_star)[0][0]
y_samples = Y[:, k]

plt.hist(y_samples, bins=80, density=True)
plt.yscale("log")
plt.title(f"Y at t={times[k]:.3f} (log y-scale)")
plt.show()


print("mean:", y_samples.mean())
print("var :", y_samples.var(ddof=1))
print("min/max:", y_samples.min(), y_samples.max())
print("quantiles (50%, 90%, 99%, 99.9%):", np.quantile(y_samples, [0.5, 0.9, 0.99, 0.999]))
print("spot for S, f):", F[:,0], S_mat[:,0])


#(py311) PS C:\Users\masao\OneDrive\Namizje\Thesis\OptStopRandNN> python -m optimal_stopping.data.model_tests
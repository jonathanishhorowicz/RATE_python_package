"""
Creates Figures 1, 2 and 3 in rate_code_validation.pdf.

Takes about 15 minutes.
"""
import numpy as np

from rate.projections import PseudoinverseProjection, CovarianceProjection
from rate.importance import RATE2
from rate.wrapped_r import init_rate_r

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

def make_plot(n_draw_vals, norms, pearsons_rho, spearmans_rho, filename):
	"""makes and saves the convergence/correlation plots"""
	fig, axes = plt.subplots(1, 2, figsize=(8,4))

	axes[0].errorbar(np.log10(n_draw_vals), norms.mean(axis=1), yerr=norms.std(axis=1), fmt='o')
	axes[0].set_xlabel("log10(number of posterior draws)")
	axes[0].set_ylabel(r"$||$R-python$||_2$")
	axes[0].set_ylim([0.0, axes[0].get_ylim()[1]])

	axes[1].errorbar(np.log10(n_draw_vals), spearmans_rho.mean(axis=1), yerr=spearmans_rho.std(axis=1), fmt='o', label="spearman")
	axes[1].errorbar(np.log10(n_draw_vals), pearsons_rho.mean(axis=1), yerr=pearsons_rho.std(axis=1), fmt='o', label="pearson")
	axes[1].legend(loc="lower right")
	axes[1].set_xlabel("log10(number of posterior draws)")
	axes[1].set_ylabel("correlation(R,python)")

	plt.tight_layout()
	fig.savefig(filename+".pdf", bbox_inches="tight")
	
#
# Settings
#
n, p = 100, 10
n_draw_vals = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
n_repeats = 10

rate_r_func = init_rate_r() # initialise R code

#
# Logit posterior - multivariate normal with full rank covariance
#
M_F = np.random.rand(n)[np.newaxis]
V_F = np.random.randn(n, n)
V_F = np.dot(V_F, V_F.T)[np.newaxis]
X = np.random.randn(n, p)

assert np.linalg.matrix_rank(V_F)==n

#####################################################################
### PSEUDOINVERSE PROJECTION - NO MATRIX FACTORISATION - FIGURE 1 ###
#####################################################################

rate_python, klds_python = RATE2(X, M_F, V_F, projection=PseudoinverseProjection(), return_KLDs=True) # the python result. Doesn't use matrix factorisation

print("Pseudoinverse projection, no matrix factorisation...", end="")

norms = np.zeros((len(n_draw_vals), n_repeats))
spearmans_rho = np.zeros((len(n_draw_vals), n_repeats))
pearsons_rho = np.zeros((len(n_draw_vals), n_repeats))

for i, n_draws in enumerate(n_draw_vals):
	for j in range(n_repeats):

		f_draws = np.random.multivariate_normal(M_F[0], V_F[0], size=(n_draws)) # Draw samples
		rate_r, klds_r, _, _ = rate_r_func(X, f_draws, "linear", False) # Calculate rate using samples (uses R code)
		norms[i,j] = np.linalg.norm(rate_r-rate_python, ord=2) # Calculate evaluation metrics (norm, correlation)
		pearsons_rho[i,j] = pearsonr(rate_r, rate_python)[0]
		spearmans_rho[i,j] = spearmanr(rate_r, rate_python)[0]

make_plot(n_draw_vals, norms, pearsons_rho, spearmans_rho, "linear_projection_validation")
print("done")

#######################################################################
### PSEUDOINVERSE PROJECTION - WITH MATRIX FACTORISATION - FIGURE 2 ###
#######################################################################

# Note - we reuse the previous Python RATE result since the matrix factorisation is only implemented in the R code

print("Pseudoinverse projection, with matrix factorisation...", end="")

norms = np.zeros((len(n_draw_vals), n_repeats))
spearmans_rho = np.zeros((len(n_draw_vals), n_repeats))
pearsons_rho = np.zeros((len(n_draw_vals), n_repeats))

for i, n_draws in enumerate(n_draw_vals):
	for j in range(n_repeats):

		f_draws = np.random.multivariate_normal(M_F[0], V_F[0], size=(n_draws))
		rate_r, klds_r, _, _ = rate_r_func(X, f_draws, "linear", True)
		norms[i,j] = np.linalg.norm(rate_r-rate_python, ord=2)
		pearsons_rho[i,j] = pearsonr(rate_r, rate_python)[0]
		spearmans_rho[i,j] = spearmanr(rate_r, rate_python)[0]

make_plot(n_draw_vals, norms, pearsons_rho, spearmans_rho, "linear_projection_validation_with_mat_fac")
print("done")

########################################
### COVARIANCE PROJECTION - FIGURE 3 ###
########################################

print("Covariance projection...", end="")

rate_python, klds_python = RATE2(X, M_F, V_F, projection=CovarianceProjection(), return_KLDs=True) # the python result

norms = np.zeros((len(n_draw_vals), n_repeats))
spearmans_rho = np.zeros((len(n_draw_vals), n_repeats))
pearsons_rho = np.zeros((len(n_draw_vals), n_repeats))

for i, n_draws in enumerate(n_draw_vals):
	for j in range(n_repeats):

		f_draws = np.random.multivariate_normal(M_F[0], V_F[0], size=(n_draws))
		rate_r, klds_r, _, _ = rate_r_func(X, f_draws, "covariance", False)
		norms[i,j] = np.linalg.norm(rate_r-rate_python, ord=2)
		pearsons_rho[i,j] = pearsonr(rate_r, rate_python)[0]
		spearmans_rho[i,j] = spearmanr(rate_r, rate_python)[0]

make_plot(n_draw_vals, norms, pearsons_rho, spearmans_rho, "cov_projection_validation")
print("done")
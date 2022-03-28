import numpy as np
import pandas as pd

from scipy.linalg import cholesky, cho_solve, cho_factor, eigh
from .solve_utils import condition_gaussian, jth_partition, kl_mvn, get_solver, chol_then_inv

import copy

from abc import ABCMeta, abstractmethod

# import sys
# sys.path.append('/home/jsi17/Documents/Dropbox/PhD/RATE/qrupdate-ng/f2py') # qrupdate
# from qrupdate import dchdex, dch1dn

import logging
logger = logging.getLogger(__name__)

class KLDSolver(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self, Mb, Vb, J, jitter=1e-9):
		self.Mb = Mb
		self.Vb = Vb + jitter * np.eye(Vb.shape[0])
		self.J = J
		self.jitter = jitter

	@abstractmethod
	def solve_all_KLDs(self):
		# given array jj of indices return KLD term plus any other info
		raise NotImplementedError

class DirectCovSolver(KLDSolver):
    def __init__(self, Mb, Vb, J, jitter=1e-9, solver_name="chol", exact=False):
        super().__init__(Mb, Vb, J, jitter)
        self.solver_name = solver_name
        self._solver = get_solver(solver_name)
        self.exact = exact
        
    def solve_all_KLDs(self, **kwargs):
        
        out_arr = np.zeros((len(self.J), 4))
        
        for out_idx, j in enumerate(self.J):

            # partition ESA posterior
            mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(
                self.Mb, self.Vb, j
            )

            # conditional distribution
            mu_cond, Sigma_cond = condition_gaussian(
                mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j
            )

            # calculate KLD
            out_arr[out_idx] = kl_mvn(
                mu_min_j, Sigma_min_j, mu_cond, Sigma_cond,
                solver=self._solver,
                jitter=self.jitter,
                exact=self.exact)
            
        out_df = pd.DataFrame(out_arr, columns=["quad", "trace", "logdet", "KLD"])
            
        return out_df, None
     
# class CholUpdateSolver(KLDSolver):
#     def __init__(self, Mb, Vb, J, jitter=1e-9):
#         super().__init__(Mb, Vb, J, jitter)
#         self.full_cholesky = cholesky(Vb)
#         logger.debug("Calculated Cholesky factor for all variables")
        
#     def solve_all_KLDs(self):
        
#         out_arr = np.zeros((len(self.J), 1))
        
#         for out_idx, j in enumerate(self.J):
            
#             # partition ESA posterior
#             mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(
#                 self.Mb, self.Vb, j
#             )

#             # conditional mean
#             mu_cond = mu_min_j - np.dot(sigma_min_j, np.linalg.lstsq(sigma_j, mu_j, rcond=None)[0])
            
#             # low-rank update to conditional covariance
#             Sigma_update = np.dot(sigma_min_j, np.linalg.lstsq(sigma_j, sigma_min_j.T, rcond=None)[0])
            
#             # update Cholesky factor by removing one row/col at a time
#             # remove them backwards so the original order is retained if
#             # more than one is removed
#             tmpchol = copy.deepcopy(self.full_cholesky)
#             for idx_var in np.flip(j):
#                 tmpchol = self._chol_del_update(tmpchol, idx_var)
                
#             # Cholesky downdate
#             logger.debug(f"Calculating {j.shape[0]} rank-one updates")
#             u_list = self._rank1_decomp(Sigma_update, j.shape[0])
#             for u in u_list:
#                 tmpchol = self._chol_downdate(tmpchol, u)

#             diff = mu_cond - mu_min_j
              
#             logger.debug(f"tmpchol.shape={tmpchol.shape}, diff.shape={diff.shape}")
#             out_arr[out_idx] = diff.T @ cho_solve((tmpchol, False), diff)

#         out_df = pd.DataFrame(
#             np.hstack([out_arr, 0.5*out_arr]), columns=["quad", "KLD"]
#         )
#         return out_df, None
        
#     def _chol_del_update(self, R, j):
#         logger.debug(f"Cholesky update for R with shape {R.shape}, j={j}")
#         # Wrapper for ``dchdex``.
#         # uses upper tringular cholesky factor

#         # Parameters
#         # ----------
#         # n : input int
#         # r : input rank-2 array('d') with bounds (ldr,*)
#         # j : input int
#         # w : input rank-1 array('d') with bounds (*)

#         # Other Parameters
#         # ----------------
#         # ldr : input int, optional
#         #     Default: shape(r,0)

#         R_copy = copy.deepcopy(R)
#         n = R_copy.shape[0]
#         w = np.zeros(n)
#         dchdex(n, R_copy, j+1, w, n) # python -> Fortran indexing
#         return R_copy[:-1,:-1]
    
#     def _chol_downdate(self, R, u):
#         '''
#         dch1dn(n,r,u,w,info,[ldr])

#         Wrapper for ``dch1dn``.

#         Parameters
#         ----------
#         n : input int
#         r : input rank-2 array('d') with bounds (ldr,*)
#         u : input rank-1 array('d') with bounds (*)
#         w : input rank-1 array('d') with bounds (*)
#         info : input int

#         Other Parameters
#         ----------------
#         ldr : input int, optional
#             Default: shape(r,0) 
#         '''
#         R_copy = copy.deepcopy(R)
#         n = R_copy.shape[0]
#         w = np.zeros(n)
#         info = 0
#         dch1dn(n, R_copy, u, w, info)
#         if info==1:
#             logger.error(f"Result is not PD")
#         elif info==2:
#             logger.error(f"Result is singular")
#         return R_copy
    
#     def _rank1_decomp(self, update_matrix, max_rank):
#         # U, s, Vh = svd(update_matrix) # computes full decomposition but we only need top k
#         s, U = eigh(update_matrix, subset_by_index=[update_matrix.shape[0]-max_rank, update_matrix.shape[0]-1])
#         return [s[k]**0.5 * U[:,k] for k in range(max_rank)]

class PrecisionSolver(KLDSolver):
    def __init__(self, Mb, Vb, J, inv_func, jitter=1e-9):
        super().__init__(Mb, Vb, J, jitter)
        self.full_precision = inv_func(Vb)
        
    def solve_all_KLDs(self):
        out_arr = np.zeros((len(self.J), 1))
        
        for out_idx, j in enumerate(self.J):
            
            # partition ESA posterior
            mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(
                self.Mb, self.Vb, j
            )
            
            # conditional mean
            mu_cond = mu_min_j - np.dot(sigma_min_j, np.linalg.lstsq(sigma_j, mu_j, rcond=None)[0])
            
            diff = mu_cond - mu_min_j
            out_arr[out_idx] = diff.T @ np.delete(np.delete(self.full_precision, j, axis=1), j, axis=0)  @ diff
            
        out_df = pd.DataFrame(
            np.hstack([out_arr, 0.5*out_arr]), columns=["quad", "KLD"]
        )
        return out_df, None
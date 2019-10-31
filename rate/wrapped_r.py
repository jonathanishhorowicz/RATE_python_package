import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import activate

import pkg_resources

def init_rate_r():
	"""Returns a python function that calls Lorin's original R code under the hood.

	The resulting function onl. uses the X, f.draws and projection arguments.

	Example usage:
		n, p = 100, 10
		n_draws = 1000
		X = np.random.randn(n, p)
		projection = "covariance"
		f_draws = np.random.multivariate_normal(np.zeros(n), np.eye(n, n), size=(n_draws))

		rate_r = init_rate_r() # Returns a function that calls the R code directly
		out = rate_r(X, f_draws, projection, False) # Final argument is for matrix factorisation	

		# out is a list of [KLDs, RATE, Delta, ESS] (i.e. matches the R function)
	"""

	### Include R string automatically - hard to get to work with pytest
	# r_code_path = pkg_resources.resource_filename(__name__, 'reference_code.R')
	# with open(r_code_path, "r") as f:
	# 	rcode = "".join(f.readlines())
	# robjects.r(rcode)

	# This long string should match reference_code.R
	robjects.r('''
	RATE <- function(X, f.draws, projection, factorize,
	  	nullify=NULL, snp.nms=colnames(X), cores=1){
	  
		### Determine the number of Cores for Parallelization ###
		if(cores > 1){
		   if(cores>parallel::detectCores()) {
		       warning("The number of cores you're setting is larger than detected cores - all available cores will be used")
		       cores = parallel::detectCores()
		   }
		}
		  
		`%dopar%` <- foreach::`%dopar%` # so that we don't need to import foreach
		cl <- parallel::makeCluster(cores)
		doParallel::registerDoParallel(cl, cores=cores)

		### Linear projection operator 
		if (projection == "linear") {

	        if(factorize) {
	            ### First Run the Matrix Factorizations, if required ### 
	            if(nrow(X) < ncol(X)){
	                #In case X has linearly dependent columns, first take SVD of X to get v.
	                # svd() returns list with d, u, v s.t. X = U D V', where d a vector of entries in D the diagonal matrix
	                svd_X = svd(X)  
	                r_X = sum(svd_X$d>1e-10)  # d is diagonal
	                u = with(svd_X,(1/d[1:r_X]*t(u[,1:r_X])))
	                v = svd_X$v[,1:r_X]
	                
	                # Now, calculate Sigma_star
	                SigmaFhat = cov(f.draws)
	                Sigma_star = u %*% SigmaFhat %*% t(u)
	                
	                # Now, calculate U st Lambda = U %*% t(U)
	                svd_Sigma_star = svd(Sigma_star)
	                r = sum(svd_Sigma_star$d > 1e-10)
	                U = t(MASS::ginv(v)) %*% with(svd_Sigma_star, t(1/sqrt(d[1:r])*t(u[,1:r])))
	            } else {
	                beta.draws = t(MASS::ginv(X)%*%t(f.draws))
	                V = cov(beta.draws); #V = as.matrix(nearPD(V)$mat)
	                D = MASS::ginv(V)
	                svd_D = svd(D)
	                r = sum(svd_D$d>1e-10)
	                U = with(svd_D,t(sqrt(d[1:r])*t(u[,1:r])))
	            }
	            Lambda = Matrix::tcrossprod(U)
	        } else {
	            # Direct inversion of the effect size analogue posterior covariance if
	            # don't want to do the factorisations
	            X.dagger = MASS::ginv(X)
	            Lambda = MASS::ginv(X.dagger%*%cov(f.draws)%*%t(X.dagger))
	        }
			  
			### Compute the Kullback-Leibler divergence (KLD) for Each Predictor ###
			mu = c(MASS::ginv(X)%*%colMeans(f.draws))
			int = 1:length(mu); l = nullify;
			  
			if(length(l)>0){int = int[-l]}
			  
			if(nrow(X) < ncol(X)){
	            KLD = foreach::foreach(j = int)%dopar%{
	            q = unique(c(j,l))
	            m = mu[q]
	            U_Lambda_sub = qr.solve(U[-q,],Lambda[-q,q,drop=FALSE])
	            kld = crossprod(U_Lambda_sub%*%m)/2
	            names(kld) = snp.nms[j]
	            kld
	            }
	        } else {
	            KLD = foreach::foreach(j = int)%dopar%{
	            q = unique(c(j,l))
	            m = mu[q]
	            alpha = t(Lambda[-q,q])%*%MASS::ginv(as.matrix(Matrix::nearPD(Lambda[-q,-q])$mat))%*%Lambda[-q,q]
	            kld = (t(m)%*%alpha%*%m)/2
	            names(kld) = snp.nms[j]
	            kld
			    }
	        }

		### Covariance projection operator 
	    } else if (projection == "covariance") {

	        if (factorize)
	            warning("Matrix factorisation is only possible with the linear/pseudoinverse projection")

	        # dim: (num. of draws in f.draws x p)
	        beta.draws <- t(cov(X, t(f.draws)))

	        # empirical mean of beta.draws
	        mu = colMeans(beta.draws)

	        int = 1:length(mu); l = nullify;
	        if(length(l)>0){int = int[-l]}

	        Lambda <- MASS::ginv(cov(beta.draws))
			      
	        ### Compute the Kullback-Leibler divergence (KLD) for Each Predictor ###
	        KLD <- foreach::foreach(j = 1:ncol(X))%dopar%{ 
	            q = unique(c(j,l))
	            m = mu[q]
	            alpha = t(Lambda[-q,q])%*%MASS::ginv(as.matrix(Matrix::nearPD(Lambda[-q,-q])$mat))%*%Lambda[-q,q]
	            kld = (t(m)%*%alpha%*%m)/2
	            names(kld) = snp.nms[j]
	            kld
	            }
			}

	    KLD = unlist(KLD)
		  
	    ### Compute the corresponding “RelATive cEntrality” (RATE) measure ###
	    RATE = KLD/sum(KLD)

	    ### Find the entropic deviation from a uniform distribution ###
	    Delta = sum(RATE*log((length(mu)-length(nullify))*RATE))

	    ### Calibrate Delta via the effective sample size (ESS) measures from importance sampling ###
	    #(Gruber and West, 2016, 2017)
	    ESS = 1/(1+Delta)*100

	    parallel::stopCluster(cl)

	    ### Return a list of the values and results ###
	    return(list("RATE"=RATE,"KLD"=KLD,"Delta"=Delta,"ESS"=ESS))
	}
	''')

	activate() # Activate R to numpy type conversions

	def wrapped_r_rate(X, f_draws, projection, factorize):
		r_out = robjects.globalenv['RATE'](X, f_draws, projection, factorize)
		return [np.array(r_out[0]), np.array(r_out[1]), np.array(r_out[2])[0], np.array(r_out[3])[0]]

	return wrapped_r_rate

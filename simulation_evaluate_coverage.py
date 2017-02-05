# This should be run in Python 3

## Double check carefulness about the seed
## Things are done in the setting where I know the value of the likelihood of the MLE but I don't observe the MLE directly
## I also only do estimation for one set of initial estimates, I could change how I initialize my quadratic, 
## and how many initial points I use

## double check gamma and normal pdfs
## double check hessian inverse stuff

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.stats import norm
from scipy.optimize import minimize
### Create storage variables for each simulation ###

## speed:Make vectors of length iter for each round so we can store all the estimates from each iteration and 
## take summary statistics at the end
true_L = []
true_U = []
true_MLE = []
L_vec = []
U_vec = []
L_vec_noise = []
U_vec_noise = []
L_vec_g_noise = []
U_vec_g_noise = []
MLE_hat = []
MLE_var = []
a_store = []
b_store = []
a_reparam_store = []
a_var_store = []
b_var_store =  []
a_reparam_var_store = []

## These will be updated additively, could have stored as a vector and taken mean, but no need
SD_g = 0
bias_g = 0

### Define optimization function to get estimate of profile likelihood ###

## This is where I define the meta model optimization function, it finds the quadratice with the highest
## likelihood for a give meta-model
## a_init,b_init,c_reparam define the initial quadratic guess
## x_star and y_star are the data values
## y_star_max is the maximum of all y_star, helpful when parameterizing c, bu could remove
## mcmc_sample_size is the number of monte carlo points you want to use to approximate the distribution
## sample_size is how many data points I have to evaluate the likelihood at
## This function works for any set of x_star, y_star but is particular to the specified distributions
def meta_model_optimization(a_init, b_init, c_reparam, x_star, y_star, x_star_sd, y_star_max, mcmc_sample_size, sample_size):
	## Define likelihood function to optimize
	def full_log_likelihood_alpha(a, b, c_reparam):
		## Set seed every time to make the mcmc sample consistent, i.e. for the same parameters, I will get the same likelihood
		## estimate
		np.random.seed(100)
		mcmc_sample = np.random.normal(-b/(2*a), np.sqrt(-1/(2*a)), mcmc_sample_size)
		temp  = a*np.square(mcmc_sample) + b*mcmc_sample + c_reparam + b**2/(4*a) + y_star_max
		cur_sum = 0
		  
		## Should I vectorize this?
		for j in range(sample_size):
			like_y_star = gamma.pdf(temp - y_star[j], a = .5, scale = 1)
			like_x_star = norm.pdf(mcmc_sample-x_star[j], loc = 0, scale = x_star_sd)
			cur_sum = cur_sum + np.log(1/mcmc_sample_size*sum(like_y_star*like_x_star))
		
		return(cur_sum)
	

	## Within this I reparameterize to allow the optimization to not have any constraints, also prevents 
	## optimization errors
	def optim_fun2(arg_vec):
		a = -np.exp(arg_vec[0])
		b = arg_vec[1]
		return(-full_log_likelihood_alpha(a, b, c_reparam))
	
	
	init_array = np.array([np.log(-a_init), b_init])
	optim_sol = minimize(optim_fun2, x0 = init_array, method = 'Nelder-Mead')
	return(optim_sol)


                                 	

### Begin Simulation ###
coverage_iter_number = 5		# number of simulations I will use to assess the coverage               
for k in range(coverage_iter_number):
	## Here I have to reset the seed because, if not, the seed reset in my optimization will mess up 
	## my data
	np.random.seed(k)

	### Create data set for this iteration and true parameter values ###
		
		## Simulation parameter
	sigma = 2*np.array([[1, .5], [.5, 3]])
	sigma_inv = np.linalg.inv(sigma)
	mu = np.array([-5.1, 5.2])
	n = 20

	## Create data

	data = np.random.multivariate_normal(mu, sigma, n)
	data_mean = np.mean(data, axis = 0)

	## Get the true profile likelihood function, need to redefine this everytime I get a new data set
	def pl_eval(theta):
		## mu1 = theta case
		mu_2_max = data_mean[1] + sigma_inv[0,1]/sigma_inv[1,1]*(data_mean[0]-theta)
		mu_2_max = min(theta, mu_2_max)
		like_1 = multivariate_normal.logpdf(np.array([theta, mu_2_max]), mean = data_mean, cov = sigma/n)

		## mu2 = theta case
		mu_1_max = data_mean[0] + sigma_inv[0,1]/sigma_inv[0,0]*(data_mean[1]-theta)
		mu_1_max = min(theta, mu_1_max)
		like_2 = multivariate_normal.logpdf(np.array([mu_1_max,theta]), mean = data_mean, cov = sigma/n)
		return(max(like_1,like_2))

	## Getting MLE of my underlying data set
	mle_sample = max(data_mean)
	true_MLE.append(mle_sample)

	## Get the likelihood value of MLE
	mle_like = multivariate_normal.logpdf(data_mean, mean = data_mean, cov = sigma/n)
	pl_eval(mle_sample) == mle_like  # should be true since the profile likelihood equals the true likelihood for the MLE

	
	## Points for plotting true profile likelihood
	theta_set = np.arange(5001)/1000 - 2.5 + mle_sample
	pl_eval_vect = np.vectorize(pl_eval)
	theta_set_like = pl_eval_vect(theta_set)
	pl_cutoff = mle_like - 1.920729

	# true confidence interval lower and upper bound, i.e. if I knew the true profile likelihood
	true_L_cur = theta_set[(theta_set_like > pl_cutoff)][0] -.0005	# lower bound
	true_U_cur = theta_set[(theta_set_like > pl_cutoff)][-1] + .0005 	  # upper bound
	true_L.append(true_L_cur)   # storage
	true_U.append(true_U_cur)	 # storage


	### Generate points 'below' the profile likelihood that I will use to estimate true profile likelihood ###
	
	## Estimation parameters
	t_g = 10			# Allotted horizontal error in each point, the larger, the smaller the horizontal error
	sample = 20		# Number of points I will generate to estimate the profile likelihood
	
	## Get my sample of points to use for estimating the profile likelihood i.e. points below the PL
	mu_sample = np.random.multivariate_normal(data_mean, sigma/n, sample) # Generate points from a normal, this distribution can be specified, for example, sigma does not have to be the same, and it doesn't have to be normal
	mu_sample_eval = np.amax(mu_sample, axis = 1)	# Get the max, i.e. x-coordinate without horizontal noise
	likehood_sample = multivariate_normal.logpdf(mu_sample, mean = data_mean, cov = sigma/n)	# Get their likelihood i.e. y coordinate

	##these are the x-coordinates with horizontal noise, could vectorize?
	mu_hat_max = np.amax(np.random.multivariate_normal(np.zeros(2), sigma/t_g, sample) + mu_sample, axis = 1)

	## This is to get an understanding of the horizontal error we introduced and how it comes
	## out in the maximum, this is not necessary for the final result, but for exploration purposes
	epsilon_vec = mu_hat_max - mu_sample_eval
	
	y_star_max = max(likehood_sample)	# Get largest likelihood value
	x_star_sd = np.std(epsilon_vec)	# Get estimate of horizontal error. Note this should change since it's based on unknown truth 
	bias_g = bias_g + sum(epsilon_vec)	# get estimate of horizontal bias
	SD_g = SD_g + x_star_sd 	# storage


	### Given my points, I get an estimate of the profile likelihood  ###
	
	## Get initial quadratic guess
	## I can alter this to get better initial estimates
	curvature = -5		# Inital estimate of curvature
	center =  np.mean(mu_hat_max)		# Initial estimate of center of my quadratic
	height = mle_like		# My height is based on the likelihood of true mle which is known
	
	## Get the corresponding values for a quadratic function
	a_init = curvature
	b_init = -2*curvature*center
	c_reparam = height - y_star_max

	## Find the optimized quadratice parameters, i.e. my PL estimate
	optimized_parameters = meta_model_optimization(a_init, b_init, c_reparam, mu_hat_max, 
		likehood_sample, x_star_sd, y_star_max, 10000, sample)
	
	a_reparam = optimized_parameters.x[0]
	a = -np.exp(a_reparam)
	b = optimized_parameters.x[1]
	#information_inv_est = optimized_parameters.hess_inv	## keep this positive since I minimized the negative log likelihood

	### Storage of values and finding new cut offs for our profile likelihood ###
	
	MLE_hat.append(-b/(2*a))		# Store estimate of MLE based on PL
	grad_mle = np.array([-b/(2*np.exp(a_reparam)), 1/(2*np.exp(a_reparam))])
	#cur_inv = information_inv_est		# get error estimates of parameters based on hessian
	#cur_MLE_var = grad_mle.dot(cur_inv).dot(grad_mle)	# get estimate of MLE variance
	#MLE_var.append(cur_MLE_var)	# store MLE variance estimate
	a_store.append(a)		# store curvaturue
	b_store.append(b)		# store b value in quadratice
	#a_reparam_store.append(a_reparam)
	#a_var_store.append(np.exp(-2*a_reparam)*cur_inv[0,0])	# variance in a estimate
	#b_var_store.append(cur_inv[1,1])	# variance in b estimate
	#a_reparam_var_store.append(cur_inv[0,0])		# variance in reparameterized a
	
	## obtain new profile likelihood cutoff based on estimated PL
	new_cut_off = y_star_max - 1.92  # Tim double check this should be y_star_max vs mle_like
	
	
	L_vec_noise_cur =  -np.sqrt((new_cut_off - (c_reparam+y_star_max)) / a ) - b/(2*a)		# New estimated lower bound
	U_vec_noise_cur =  np.sqrt((new_cut_off - (c_reparam+y_star_max)) / a ) - b/(2*a)		# New estimated upper bound
	
	L_vec_noise.append(L_vec_noise_cur)
	U_vec_noise.append(U_vec_noise_cur) 


	### Create a plots to check out the results while it's running ###
	
	## Plot the true profile likelihood in blue
	#plot(theta_set, theta_set_like, cex = .02, xlim = c(mle_sample-2.3,mle_sample+2.3), ylim = c(mle_like-3, mle_like+.3), col=4, xlab = "Theta", ylab = "Likelihood", main = "Sim2-mu_2, small MCMC sample")
	#points(mle_sample,mle_like, col=4, cex = .3)	## plot the mle
	
	## plot cut off and lower and upper bounds for true PL
	#abline(h = pl_cutoff)
	#points(true_L_cur, pl_cutoff, col = 4)
	#points(true_U_cur, pl_cutoff, col = 4)
	
	## Plot the initial estimate of the profile likelihood in turqouise
	#quad_fun = function(x) {return(a_init*(x+b_init/(2*a_init))**2 + c_reparam + y_star_max)}
	#points(theta_set, quad_fun(theta_set),col=5, cex = .02)
	
	## Plot the optimized PL estimate in pink
	#quad_fun = function(x) {return(a*(x+b/(2*a))**2 + c_reparam+y_star_max)}
	#points(theta_set, quad_fun(theta_set),col=6, cex = .02)
	
	## Plot the new cut off and lower and bounds for estimated PL
	#abline(h = new_cut_off)
	#abline(v = max(mu) )
	#points(L_vec_noise_cur, new_cut_off, col = 4)
	#points(U_vec_noise_cur, new_cut_off, col = 4)
	
	## Print the iteration
	print(k)


### Check out results ###

MLE_hat
MLE_var
a_store 
b_store
a_reparam_store
#a_var_store
#b_var_store
#a_reparam_var_store

# Compare variance of MLE_hat to estimated variance of MLE_hat
var(MLE_hat-true_MLE)
mean(MLE_var)

## Bias of MLE_hat
mean(MLE_hat-true_MLE)

#which(!(true_U > 5.2 & true_L < 5.2))

#which(!(U_vec_noise > 5.2 & L_vec_noise < 5.2))

#sum(np.logical_and(true_U > 5.2, true_L < 5.2))
#sum(np.logical_and(U_vec_noise > 5.2, L_vec_noise < 5.2))

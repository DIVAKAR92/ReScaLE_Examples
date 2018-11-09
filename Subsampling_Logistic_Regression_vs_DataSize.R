#=========================================================================================
#       Simulate data to run the logistic regression model 
#=========================================================================================
library(MASS) 
# mean vecctor and the covariance matrix to simulate the normal random variable
mu <- c(0)
Sigma <- diag(1,1,1)

# I have set intercept term in the logistic model to -0.5
beta.0 <- -0.5 

# set the number of observations to simulate 
n.obs <- 1000000
# setting the seed 
set.seed(1)
# simulate the normal random variables which acts as our explanatory variable
x <- rnorm(n.obs, mu, Sigma)
# find the minimum and the maximum of the simulated data points
m = min(x); M = max(x)
# we subtract the minimum and divide by the range
x = (x-m)/(M-m)
# set beta_1 to be equal to 1
beta_1 <- c(1)   
# set the seed and simulate the response variable 
set.seed(1)
y <- as.double(runif(n.obs) < 1 / (1 + exp(-beta.0 - x * beta_1)))
sum(y)

#=========================================================================================
#       Run the logistic regression model and extract control-variate information
#=========================================================================================
# prepare the data as a data frame and run a logistic regression model
X = data.frame(Y =y, X0=x)
# logistic regression model
logit_model = glm(Y~X0,data=X,family = binomial())

# extract the variance-covariance matrix
V = vcov(logit_model)
# perform Cholesky decomposition to get the square-root of the variance-covariance matrix
sd = t(chol(V))
# extract the MAP value of the posterior distribution
map = coef(logit_model)

#=========================================================================================
#       Prepare the data & other information and send to the C++ program
#=========================================================================================
Y = X$Y
X0 = (X$X0)
n = length(Y)

# gradient of the log-likelihood function 
# this is needed to calculate the constant C
grad_log_phi = function(x){
  # transform the point x 
  x = (sd%*%x) + map
  # probabilities corresponding to each data point
  p = 1/(1 + exp(-x[1]-X0*x[2] ))
  res = x
  # full gradient of the loglikelihood function
  res[1] = sum((Y-p));
  res[2] = sum((Y-p)*X0);
  res
}

# gradient value at MAP value under the transformation 
grad = t(sd)%*%grad_log_phi(c(0,0))

# second derivative matrix of the log-likelihood function 
div_log_phi = function(x){
  # transformation of the point x
  x = (sd%*%x) + map
  # probabilities corresponding to each data point 
  p = 1/(1 + exp(-x[1]-X0*x[2] ))
  # calculation of the second derivative matrix 
  res= matrix(0,2,2)
  res[1,] = c(- sum(p*(1-p)),    - sum(p*(1-p)*X0)) 
  res[2,] = c(- sum(p*(1-p)*X0), - sum(p*(1-p)*X0*X0))
  res
}

# computing the divergence under the tranformation 
div = sum(diag(t(sd)%*%div_log_phi(c(0,0))%*%sd))

# calculation of the constant C
C = (sum(grad^2) + (div) )/(2)
# 2 times the gradient of the log-likelihood 
tgl = 2*grad
C 

# get a rough estimate of P_n, I change it manually to get the better bounds 
# I change the value of P_n manually in the main code, the relevant functions
# are kappa__ and M__.
mat = matrix(1,2,2)
mat = t(sd)%*%mat%*%sd
p_n = max(eigen(mat)$value)
p_n

#=========================================================================================
#       send all the information in the main code (data, control variates and 
#       transformation matrix)
#=========================================================================================
# the data size
N = length(Y)
# control variates and related information
store_constants(c(map,N, C, tgl))
# the transformation matrix
store_TM(c(sd[1,1],sd[2,1],sd[2,2]))
# the data 
store_data_2(X0,Y)


#=========================================================================================
#       Run the ReScaLE algorithm and transform back its output
#=========================================================================================
# running the algorithm 
tmp =system.time({data = convert_to_dataframe(10000,rep(0,2) ,2,L=0.1,prior_C = 0, Lambda = 0.0)}) 
# transforming factor set to 1
tf = sqrt(1)
# apply the matrix of transformation on the skeleton points (for simplicity)
y1 = (sd[1,1]*data$x1 + sd[1,2]*data$x2 )/tf + map[1]
y2 = (sd[2,1]*data$x1 + sd[2,2]*data$x2 )/tf + map[2]
# get the density and plot it
d = density(y1)
plot(d, col=1, lwd=1, ylim=c(0,50))
# overlay the Bernstein-von-Mises Normal approximation
i = 1
lines(d$x, dnorm(d$x,map[i], sqrt(V[i,i])), col=2)


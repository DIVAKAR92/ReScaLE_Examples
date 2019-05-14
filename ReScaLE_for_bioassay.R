#======================================================================================
#                Run the ReScaLE C++ code for the bioassay Data
#======================================================================================

Rcpp::sourceCpp('bioassaylogistic_layered_rescale.cpp')

#======================================================================================
#                Logistic Regression Model on the bioassay Data
#======================================================================================
# Normalize the data (subtract the mean and divide by the sd)
X = c(-0.56047741,-0.13633234  ,0.05301813,  0.64379162)

bioassay = data.frame(x = X, 
                      N = rep(5, 4), n = c(0, 1, 3, 5))
# Run a logistic regression model 
glm_model = glm(cbind(n, N - n) ~ x, 
                data=bioassay, family =binomial)
# summary of the logistic regression model 
summary(glm_model)

# extract the estimated coefficients 
map = as.vector(coef(glm_model))
# estimate of the variance covariance matrix 
V = vcov(glm_model)
# Devise the transformation matrix which will be used to transform the posterior 
L = t(chol(V))

# Here I have set the map and the preconditioning matrix to some adhoc values intensionally
# based on the above estimate of map and L

map = c(-0.15348 ,  9.23450)
L[1, ] = c(0.7118026,0)
L[2, ] = c(0.5456446, 5.253438)

#======================================================================================
#                Bahviour of Kappa based on the bioassay Data
#======================================================================================
x = rep(0, 100000)
y = x
for(i in 1:length(x)){
  tmp = runif(2, -1,1)
  y[i] = bioassay_kappa_trans(tmp)
}
summary(y)

#======================================================================================
#                Run the ReScaLE Algorithm on the Menarche Data
#======================================================================================
# Let us start the algorithm at the posterior mode 
start = c(0,0)
# run the rescale algorithm upto a given maximum diffusion time 
data <- convert_to_dataframe(1000000,start,2,Lambda=0, prior_C = 0)

# create a time mesh to get the position of killed Brownian motion
# mesh <- seq(0,1000000, length.out = 1000000)
# x1 <- skeleton_at_given_mesh(data$t,data$x1,data$L1,data$U1,data$tau,data$W_tau1,mesh)
# x2 <- skeleton_at_given_mesh(data$t,data$x2,data$L2,data$U2,data$tau,data$W_tau2,mesh)

# transform back the ReScaLE run
tf = sqrt(100)
y1 = (L[1,1]*data$x1 + L[1,2]*data$x2)/tf + map[1]
y2 = (L[2,1]*data$x1 + L[2,2]*data$x2)/tf + map[2]

# posterior density of beta_0
d = density(y1)
# plot the posterior density of beta_0
plot(d, col=1)
# plot the Bernstein-von-Mises aprroximation
lines(d$x,dnorm(d$x,map[1],sqrt(V[1,1])), col=2)

#========================== MCMClogit under prior ===============================
# a function for the prior density 
user.prior.density = function(x){
  logden = log(dcauchy(x[1],0,10)) + log(dcauchy(x[2],0,2.5))
  logden
}

# reorganize and elongate the data to run the MCMClogit function
y = c(rep(0,5), rep(1,1),rep(0,4), rep(1, 3),rep(0,2), rep(1,5))
new_x = c(rep(X[1],5), rep(X[2],5), rep(X[3], 5), rep(X[4],5))

# Run the MCMClogit function
logit_mcmc = MCMCpack::MCMClogit(y~new_x,user.prior.density = user.prior.density,
                                 mcmc = 5000000)

# Extract the posterior sample matrix
post_samp = as.matrix(logit_mcmc)
# plot the density of beta parameter
i = 2
d = (density(post_samp[,i]))
plot(d)
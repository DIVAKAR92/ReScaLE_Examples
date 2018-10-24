#======================================================================================
#                Run the ReScaLE C++ code for the Menarche Data
#======================================================================================

Rcpp::sourceCpp('Menarche_Logistic_Regression_Experiment.cpp')

#======================================================================================
#                Logistic Regression Model on the Menarche Data
#======================================================================================
library(MASS)
attach(menarche)
X = c()
Y = c()
for(i in 1:nrow(menarche)){
  X = c(X,rep(menarche$Age[i], times = menarche$Total[i]))
  Y = c(Y, rep(1, times = menarche$Menarche[i]))
  Y = c(Y, rep(0, times = menarche$Total[i]- menarche$Menarche[i]))
}
# Normalize the data (subtract the mean and divide by the sd)
X = (X-mean(X))/sd(X)
# Run a logistic regression model 
menarche_logit = glm(Y~X, family = binomial())
# summary of the logistic regression model 
summary(menarche_logit)
# extract the estimated coefficients 
map = as.vector(coef(menarche_logit))
# estimate of the variance covariance matrix 
V = vcov(menarche_logit)
# Devise the transformation matrix which will be used to transform the posterior 
L = t(chol(V))

#======================================================================================
#                Bahviour of Kappa based on the Menarche Data
#======================================================================================
x = rep(0, 100000)
y = x
for(i in 1:length(x)){
  tmp = runif(2, -1,1)
  y[i] = menarche_kappa_trans(tmp)
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
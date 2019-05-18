# ================  source the airline code file =====================================
Rcpp::sourceCpp('trans_gunawan_sec_app_layered_rescale.cpp')
#=====================================================================================

# ================  load the data file and extract  ==================================
Airline = load('Airline.RData')
Airline = Airline[,-1]
Y  = Airline$V2
X0 = Airline$V3
X1 = Airline$V4
X2 = Airline$V5
#=====================================================================================

# ================  pre-estimated MAP and transformation matrix ======================
map = c(-1.5608909, -0.1698398,  0.2823297,  0.9864787)
n = length(Y)
sd = matrix(0,4,4)
sd[1,] = c( 0.0004034602,  0.000000e+00,  0.0000000000, 0.000000000)
sd[2,] = c(-0.0001688100,  5.071868e-04,  0.0000000000, 0.000000000)
sd[3,] = c(-0.0001442231, -4.315574e-05,  0.0006775354, 0.000000000)
sd[4,] = c(-0.0014214838, -5.157533e-04, -0.0003705154, 0.001223097)
#=====================================================================================

# ================  control-variate calculations =====================================
grad_log_phi = function(x){
  x = (sd%*%x) + map
  p = 1/(1 + exp(-x[1]-X0*x[2] - X1*x[3] - X2*x[4] ))
  res = x
  res[1] = sum((Y-p));
  res[2] = sum((Y-p)*X0); res[3] = sum((Y-p)*X1); res[4] = sum((Y-p)*X2);
  res
}

grad = t(sd)%*%grad_log_phi(c(0,0,0,0))

div_log_phi = function(x){
  x = (sd%*%x) + map
  p = 1/(1 + exp(-x[1]-X0*x[2] - X1*x[3] - X2*x[4] ))
  res= matrix(0,4,4)
  res[1,] = c(- sum(p*(1-p)),    - sum(p*(1-p)*X0), - sum(p*(1-p)*X1),    - sum(p*(1-p)*X2))
  res[2,] = c(- sum(p*(1-p)*X0), - sum(p*(1-p)*X0), - sum(p*(1-p)*X0*X1), - sum(p*(1-p)*X0*X2))
  res[3,] = c(- sum(p*(1-p)*X1), - sum(p*(1-p)*X1*X0), - sum(p*(1-p)*X1), - sum(p*(1-p)*X1*X2))
  res[4,] = c(- sum(p*(1-p)*X2), - sum(p*(1-p)*X2*X0), - sum(p*(1-p)*X2*X1), - sum(p*(1-p)*X2*X2))
  res
}

div = diag(t(sd)%*%div_log_phi(c(0,0,0,0))%*%sd)

grad = rep(0,4)
C = (sum(grad^2) + sum(div) )/(2)
tgl = 2*grad

#=====================================================================================

# ================  storing all information  =========================================

N = length(Y)
store_constants(c(map,N, C, tgl))
store_data_2(X0,X1,X2,Y)

#=====================================================================================

# ================  run the rescale method  ==========================================
# run the rescale method
data = convert_to_dataframe(10000000,rep(0,4),4,prior_C = 0)

# get the positions at a given time mesh 
mesh <- seq(0,1000000, by = 1)
f2 = dim(data)[1]/max(data$t)
x1 <- skeleton_at_given_mesh_new_2(data$t,data$x1,data$L1,data$U1,data$tau,data$W_tau1,mesh,f2 = f2, f1=15000)
x2 <- skeleton_at_given_mesh_new_2(data$t,data$x2,data$L2,data$U2,data$tau,data$W_tau2,mesh,f2 = f2, f1=15000)
x3 <- skeleton_at_given_mesh_new_2(data$t,data$x3,data$L3,data$U3,data$tau,data$W_tau3,mesh,f2 = f2, f1=15000)
x4 <- skeleton_at_given_mesh_new_2(data$t,data$x4,data$L4,data$U4,data$tau,data$W_tau4,mesh,f2 = f2, f1=15000)

# Transform back each variables
tf = 100
y1 = (sd[1,1]*x1 + sd[1,2]*x2 + sd[1,3]*x3 + sd[1,4]*x4)/sqrt(tf) + map[1]
y2 = (sd[2,1]*x1 + sd[2,2]*x2 + sd[2,3]*x3 + sd[2,4]*x4)/sqrt(tf) + map[2]
y3 = (sd[3,1]*x1 + sd[3,2]*x2 + sd[3,3]*x3 + sd[3,4]*x4)/sqrt(tf) + map[3]
y4 = (sd[4,1]*x1 + sd[4,2]*x2 + sd[4,3]*x3 + sd[4,4]*x4)/sqrt(tf) + map[4]

#=====================================================================================

# ================  run the RWM on airline data ======================================

# running mcmclogit function for a maximum diffusion time of 10000
postr = MCMCpack::MCMClogit(Y ~ X0 + X1 + X2, mcmc = 10000)


















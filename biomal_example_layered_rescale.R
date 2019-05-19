#======================================================================================
#                Run the ReScaLE C++ code for the bimodal example
#======================================================================================

Rcpp::sourceCpp('biomal_example_layered_rescale.cpp')

#======================================================================================
#                Actual target distribution for the bimodal example
#======================================================================================

x = seq(-7,7, length.out = 10000)
y = 0.33*dnorm(x, -2.5) + 0.67*dnorm(x,2.5)

#==========================================================================================
#                Run the ReScaLE method for the bimodal example (without any head start)
#=========================================================================================
# maximum diffusion time set for this example
max_time = 100000
# get a ReSCaLE run
data = convert_to_dataframe(max_time,c(0),1,L=0.1, prior_C = 0, Lambda = 0.0)
# An independent time mesh
mesh <- seq(0,max_time, by = 1)
f2 = dim(data)[1]/max(data$t)
# generate the positions at the independent time mesh
x1 <- skeleton_at_given_mesh_new_2(data$t,data$x1,data$L1,data$U1,data$tau,data$W_tau1,mesh,f2 = f2, f1=1500)
# Get the density 
d = density(x1)
# plot the actual density against the 
plot(x, y, type = 'l', col=2)
lines(d, col=1)

#==========================================================================================
#                Run the ReScaLE method for the bimodal example (uniform head start)
#=========================================================================================
# set the threshold value for the influence of the uniform prior,  for example
r = 500
# maximum diffusion time set for this example
max_time = 100000
# get a ReSCaLE run
data = convert_to_dataframe(max_time, c(0), 1, L=0.1, prior_C = r, Lambda = 0.0)
# An independent time mesh
mesh <- seq(0,max_time, by = 1)
f2 = dim(data)[1]/max(data$t)
# generate the positions at the independent time mesh
x1 <- skeleton_at_given_mesh_new_2(data$t,data$x1,data$L1,data$U1,data$tau,data$W_tau1,mesh,f2 = f2, f1=1500)
# Get the density 
d = density(x1)
# plot the actual density against the 
plot(x, y, type = 'l', col=2)
lines(d, col=1)


#==========================================================================================
#                Run the ReScaLE method for the bimodal example (bimodal head start)
#=========================================================================================
# uncomment the line reg_val[i] = 0.3*rnorm(1,-2.5,2)[0] + 0.7*rnorm(1,2.5,2)[0]; 
# can be found in the .cpp file and then comment other lines there 765-768
Rcpp::sourceCpp('biomal_example_layered_rescale.cpp')

# set the threshold value for the influence of the uniform prior,  for example
r = 500
# maximum diffusion time set for this example
max_time = 100000
# get a ReSCaLE run
data = convert_to_dataframe(max_time, c(0), 1, L=0.1, prior_C = r, Lambda = 0.0)
# An independent time mesh
mesh <- seq(0,max_time, by = 1)
f2 = dim(data)[1]/max(data$t)
# generate the positions at the independent time mesh
x1 <- skeleton_at_given_mesh_new_2(data$t,data$x1,data$L1,data$U1,data$tau,data$W_tau1,mesh,f2 = f2, f1=1500)
# Get the density 
d = density(x1)
# plot the actual density against the 
plot(x, y, type = 'l', col=2)
lines(d, col=1)

#==========================================================================================
#                Run the ReScaLE method for the bimodal example (target as the head start)
#=========================================================================================
# uncomment the line reg_val[i] = 0.3*rnorm(1,-2.5,1)[0] + 0.7*rnorm(1,2.5,1)[0]; 
# can be found in the .cpp file and then comment other lines there 765-768
Rcpp::sourceCpp('biomal_example_layered_rescale.cpp')

# set the threshold value for the influence of the uniform prior,  for example
r = 500
# maximum diffusion time set for this example
max_time = 100000
# get a ReSCaLE run
data = convert_to_dataframe(max_time, c(0), 1, L=0.1, prior_C = r, Lambda = 0.0)
# An independent time mesh
mesh <- seq(0,max_time, by = 1)
f2 = dim(data)[1]/max(data$t)
# generate the positions at the independent time mesh
x1 <- skeleton_at_given_mesh_new_2(data$t,data$x1,data$L1,data$U1,data$tau,data$W_tau1,mesh,f2 = f2, f1=1500)
# Get the density 
d = density(x1)
# plot the actual density against the 
plot(x, y, type = 'l', col=2)
lines(d, col=1)

# ==================  function for plotting the broken rescale chains =================================

plot_segments <- function(data, start, end, plot.layer = 0){
  data_sub <- subset(data, data$t <= end)
  data_sub <- subset(data_sub, data_sub$t >= start)
  xlm      <- c(start, end)
  ylm      <- c(min(data_sub$x), max(data_sub$x))
  # par(mar = c(5, 4, 1.4, 0.2))
  plot(1, 2, type = 'n', xlim = xlm, ylim = ylm, 
       xlab = 'Time',ylab= expression(X[t]), main = ' ', axes = FALSE,cex.lab=0.75 )
  u <- par("usr") 
  library(shape)
  Arrows(u[1], u[3], u[2], u[3], code = 2, xpd = TRUE) 
  Arrows(u[1], u[3], u[1], u[4], code = 2, xpd = TRUE)
  axis(1, cex.axis=0.5)
  axis(2, cex.axis=0.5)
  kill_id  <- which(data_sub$pty == 13)
  start_id <- kill_id + 2
  kill_id  <- c(kill_id)
  start_id <- c(1, start_id[1:(length(start_id)-1)])
  N <- min(length(start_id),length(kill_id))
  for(i in 1:(N-1)){
    lines(data_sub$t[(start_id[i]):kill_id[i]],
          data_sub$x[(start_id[i]):kill_id[i]], lwd=0.1, cex=0.3, lty=1)
    points(data_sub$t[(start_id[i])], data_sub$x[(start_id[i])],
           pch = 19, cex =0.3, col=3)
    points(data_sub$t[(kill_id[i])], data_sub$x[(kill_id[i])],
           pch = 19, cex =0.3, col=2)
  }
  # points(data_sub$t[start_id], data_sub$x[start_id], pch = 19, cex =0.3, col=3)
  # poi_id <- which(data_sub$pty == 1)
  # points(data_sub$t[poi_id], data_sub$x[poi_id],pch = 2, cex =0.3, col=2)
  #points(data_sub$t[kill_id], data_sub$x[kill_id],pch = 19, cex =0.3, col=2)
  legend("topright",pch=c(19),pt.cex=0.5,col = c(3),cex = 0.75,
         legend = c("Point of regeneration"),
         bty="n")
  legend("bottomright",pch=c(19),pt.cex=0.5,col = c(2),cex = 0.75,
         legend = c("Point of kill"),
         bty="n")
}

#==========================================================================================
#                Plot the figure as presented in the thesis
#=========================================================================================

posterior1 = function(x, l1=-2.5, l2=2.5){
  y=0.33*dnorm(x, l1) + 0.67*dnorm(x,l2)
  y
}

par("mar" = c(4.0,1.5,1.5,1.5))
layout(matrix(c(1,2),byrow = T,ncol=2),widths  = c(1.5,5.5))
data2 = as.data.frame(data)
m = min(data2$x); M = max(data2$x)
d = density(data2$x1[data2$t > 0])
plot(-d$y, d$x, type = 'l', axes = F, ylim = c(m,M),xlab = 'Density',cex.lab=0.75, 
     xlim=c(-max(d$y, posterior1(d$x)),0))
x = seq(m, M, length.out = 10000)
lines(-(posterior1(x)), x, col=2)
lines(-rep(0.05,times = length(x)), x, col=3)
#lines(-(0.3*dnorm(x,-2.5,2)+0.7*dnorm(x,2.5,2)),x,col=3)
u <- par("usr") 
library(shape)
Arrows(u[2], u[3],u[2], u[4], code = 3, xpd = TRUE) 
legend('topleft', col=c(1,2,3), legend=c("ReScaLE", "Truth", 
                                         expression(paste(pi[0]))), 
       lwd=2, bty='n', cex=0.50)
plot_segments(data2, 0,50000)




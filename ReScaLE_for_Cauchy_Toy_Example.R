#======================================================================================
#                Run the ReScaLE C++ code for the Cauchy Toy Example
#======================================================================================
Rcpp::sourceCpp('Cauchy_Toy_Example_Bounded_Hazard.cpp')

#======================================================================================
#                Bahviour of Kappa for the Cauchy Toy Example
#======================================================================================
x = seq(-10,10, length.out = 10000)
y = x
for(i in 1:length(x)){
  y[i] = kappa(x[i])
}
# plot the non-negative hazard rate i.e. kappa 
plot(x, y, type = 'l', xlab = "x", ylab = expression(kappa(x)))

#======================================================================================
#                Run the ReScaLE algorithm for the Cauchy Toy Problem
#======================================================================================
# note that here we specify the upper bound of kappa as an argument 
data = rescale_skeleton(50000,0,M = 14)
# use the skeleton points for simplicity 
d = density(data$x)
plot(d, col=1, main = "", xlab = expression(x))
# overlay the numerically approximated posterior density 
lines(d$x, posterior(d$x), col=2)
# set the legend 
legend("topright", legend = c("ReScaLE", "Approx"), col=c(1,2), lwd=2, bty = "n")

#======================================================================================
#                Visualising the ReScaLE segments
#======================================================================================
plot_segments <- function(data, start, end){
  data_sub <- subset(data, data$t <= end)
  data_sub <- subset(data_sub, data_sub$t >= start)
  xlm      <- c(start, end)
  ylm      <- c(min(data_sub$x), max(data_sub$x))
  # par(mar = c(5, 4, 1.4, 0.2))
  plot(1, 2, type = 'n', xlim = xlm, ylim = ylm, 
       xlab = 'time', ylab= 'Positions', main = 'Segment Plot' )
  start_id <- which(data_sub$pty == 0)
  kill_id  <- which(data_sub$pty == 13)
  kill_id  <- c(0,kill_id,nrow(data_sub))
  N = min(length(start_id),length(kill_id))
  for(i in 1:(N-1)){
    lines(data_sub$t[(kill_id[i]+1):kill_id[i+1]],
          data_sub$x[(kill_id[i]+1):kill_id[i+1]], lwd=0.2)
  }
  points(data_sub$t[start_id], data_sub$x[start_id],
         pch = 19, cex =0.3, col=3)
  points(data_sub$t[kill_id], data_sub$x[kill_id],
         pch = 19, cex =0.3, col=2)
}
# plot the segments 
plot_segments(data,0, 5)

#======================================================================================
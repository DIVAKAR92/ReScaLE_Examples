# Function to plot the bivariate movement of the ReScaLE segments
biv_plot_segments_mult <- function(data, start=0, end=max(data$t), map = c(data$x1[1], data$x2[1]),
                                   add.p=TRUE, plot.layer = 0, add = 0,...){
  data_sub <- subset(data, data$t <= end)
  data_sub <- subset(data_sub, data_sub$t >= start)
  xlm      <- c(start, end)
  positions1 = data_sub[,(3+1)]
  positions2 = data_sub[,(3+2)]
  y1 = positions1[1]
  y2 = positions2[1]
  xlm = c(min(positions1), max(positions1))
  ylm      <- c(min(positions2), max(positions2))
  # par(mar = c(5, 4, 1.4, 0.2))
  if(add == 0){
    plot(1, 2, type = 'n', xlim = xlm, ylim = ylm, axes=FALSE,
         xlab = expression(beta[0]),ylab = expression(beta[1]),
         cex.axis=0.7, cex.lab=0.7 )
  }
  u <- par("usr") 
  library(shape)
  Arrows(u[1], u[3], u[2], u[3], code = 2, xpd = TRUE) 
  Arrows(u[1], u[3], u[1], u[4], code = 2, xpd = TRUE)
  axis(1, cex.axis=0.5)
  axis(2, cex.axis=0.5)
  start_id <- which(data_sub$pty == 0)
  kill_id  <- which(data_sub$pty == 13)
  kill_id  <- c(0,kill_id,nrow(data_sub))
  N <- min(length(start_id),length(kill_id))
  for(i in 1:(N-1)){
    lines(positions1[(kill_id[i]+1):kill_id[i+1]],
          positions2[(kill_id[i]+1):kill_id[i+1]], lwd=0.1,...)
    points(positions1[(kill_id[i]+1)],positions2[(kill_id[i]+1)], col=3, 
           pch=19,cex=0.1)
    points(positions1[(kill_id[i+1])],positions2[(kill_id[i+1])], col=2, 
           pch=19,cex=0.1)
  }
  if(add.p == TRUE){
    points(y1[1], y2[1], pch = 24, cex=2, col="blue", bg="red", lwd=2)
    points(map[1],map[2],pch=23,cex=2,col="blue", bg="red", lwd=2) 
    legend("bottomright",pch=c(24,23),pt.cex=2,col = c("blue","blue"),cex = 0.75,
           pt.bg =c("red","red"),legend = c("Initial Position", "Posterior Mode"),
           bty="n")
    legend("topleft",pch=c(19,19),pt.cex=0.5,col = c(3,2),cex = 0.75,
           legend = c("Point of regeneration", "Point of kill"),
           bty="n")
  }
}


# Function to plot the ReScaLE segments corresponding to a dimension 
plot_segments_mult <- function(data, start, end,dim =1, plot.layer = 0, add = 0,...){
  data_sub <- subset(data, data$t <= end)
  data_sub <- subset(data_sub, data_sub$t >= start)
  xlm      <- c(start, end)
  positions = data_sub[,(3+dim)]
  ylm      <- c(min(positions), max(positions))
  # par(mar = c(5, 4, 1.4, 0.2))
  ylab <- bquote(beta[.(dim-1)][t])
  if(add == 0){
    plot(1, 2, type = 'n', xlim = xlm, ylim = ylm, 
         xlab = 'Time (t)',ylab = ylab, axes = FALSE,cex.axis=0.7, cex.lab=0.7)
  }
  u <- par("usr") 
  library(shape)
  Arrows(u[1], u[3], u[2], u[3], code = 2, xpd = TRUE) 
  Arrows(u[1], u[3], u[1], u[4], code = 2, xpd = TRUE)
  axis(1, cex.axis=0.5)
  axis(2, cex.axis=0.5)
  start_id <- which(data_sub$pty == 0)
  kill_id  <- which(data_sub$pty == 13)
  kill_id  <- c(0,kill_id,nrow(data_sub))
  N <- min(length(start_id),length(kill_id))
  for(i in 1:(N-1)){
    lines(data_sub$t[(kill_id[i]+1):kill_id[i+1]],
          positions[(kill_id[i]+1):kill_id[i+1]], lwd=0.2,...)
    points(data_sub$t[(kill_id[i]+1)],positions[(kill_id[i]+1)], col=3, 
           pch=19,cex=0.2)
    points(data_sub$t[(kill_id[i+1])],positions[(kill_id[i+1])], col=2, 
           pch=19,cex=0.2)
  }
  legend("bottomright",pch=c(19),pt.cex=0.5,col = c(3),cex = 0.75,
         legend = c("Point of regeneration"),
         bty="n")
  legend("bottomleft",pch=c(19),pt.cex=0.5,col = c(2),cex = 0.75,
         legend = c("Point of kill"),
         bty="n")
}

# Function to calculate the average life time of a ReScaLE chain
life_time_avg = function(data){
  kill_id = which(data$pty == 13)
  start_id = c(1,kill_id + 1)
  start_id = start_id[-length(start_id)]
  life = mean(data$t[kill_id]-data$t[start_id])
  life
}

# Function to compute the uniform norm distance with respect to a Normal distribution
unif_norm_dist_norm = function(sample1, mu, sd, x){
  P = ecdf(sample1)
  p = P(x)
  q = pnorm(x, mu, sd)
  max(abs(p-q))
}

# Function to compute the uniform norm distance between two samples from the same distribution
# at a sepecified set of points x

unif_norm_dist = function(sample1, sample2, x){
  P = ecdf(sample1)
  p = P(x)
  Q = ecdf(sample2)
  q = Q(x)
  max(abs(p-q))
}



data<-read.delim2("P:/GPRD/SSLDA/Bayes/Development/output/final-lambda.dat")
lambdadata<-as.matrix(as.vector(data),nrow=356482,ncol=60, byrow=TRUE)

ihat <- build.graph(0.02 , lambdadata, and=T)

library

x = scale(x)
p = ncol(x)
n = nrow(x)
Shat = matrix(F,p,p)

#cat("n=",n," p=",p, " lambda=",lambda,"\n", sep="")
for (j in 1:p) {
  cat(".")
  if (j %% 10 == 0) {
    cat(j)
  }
  # The response is the j-th column
  y = x[,j]
  X = x[,-j]
  
  # Do the l1-regularized regression
  # Note: the bound in l1ce code is the upper bound on the l1
  # norm.  So, a larger bound is a weaker constraint on the model
  data = data.frame(cbind(y,X))
  out = l1ce(y ~ X, data=data, sweep.out = ~1, bound=lambda)
  
  indices = (1:p)[-j]
  beta = coef(out)[2:p] # skipping the intercept
  nonzero = indices[beta > 0]
  Shat[j,nonzero] = T
  Shat[j,j] = T
}
cat("\n")

# Include an edge if either (and=F) or both (and=T) endpoints are neighbors
Ihat = matrix(F,p,p)
if (and==T) {
  for (i in 1:p) {
    Ihat[,i] = Shat[,i] & Shat[i,]
  }
}
else {
  for (i in 1:p) {
    Ihat[,i] = Shat[,i] | Shat[i,]
  }      
}
image(Ihat,col=heat.colors(2),xaxp=c(-1,2,1),yaxp=c(-1,2,1))
title(main = "Estimated graph")

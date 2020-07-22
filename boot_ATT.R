require(MatchIt)
require(foreach)
require(doParallel)

n = 1000
M = 1000
B = 1000
tau = 2
alpha = 0.3
X = runif(1000)
A = (runif(n)<(alpha/(1+alpha)))
Y = rnorm(n,0,1)
Y[A==TRUE] = tau
treat_data = data.frame(A,X,Y)
upper_quantile = vector(length = 1000)
lower_quantile = vector(length = 1000)
for (i in 1:1000){
  cl = makeCluster(4)
  registerDoParallel(cl)
  ATT_boot = foreach(i = 1:B, .combine = cbind, .packages = 'MatchIt')%dopar%{
    boot_samp = sample(seq(1,1000),1000,replace = TRUE)
    treat_data_boot = treat_data[boot_samp,]
    rownames(treat_data_boot) = seq(1,1000)
    match = matchit(A~X,distance = 'mahalanobis',method='nearest',data = treat_data_boot, replace = TRUE)
    a = as.matrix(match$match.matrix)
    ATT = mean(treat_data_boot$Y[as.integer(rownames(a))]-treat_data_boot$Y[as.integer(a[,1])])
    return(ATT)
  }
  stopCluster(cl)
  upper_ATT = as.numeric(quantile(ATT_boot,0.975))
  lower_ATT = as.numeric(quantile(ATT_boot,0.025))
  upper_quantile[i] = upper_ATT
  lower_quantile[i] = lower_ATT
}

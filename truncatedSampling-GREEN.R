
truncated_sampling <- function(mean, covariance, lower,upper, num_samples){
  library(tmvtnorm)
  library(matrixcalc)
  library(corpcor)
  library(gmm)

  mean_vector = c(mean)
  mean_vector = sapply(mean_vector, as.numeric)
  mean_vector = as.vector(mean_vector)

  covariance_matrix = as.matrix(sapply(covariance, as.numeric))
  covariance_matrix = t(covariance_matrix) %*% covariance_matrix #https://mathworld.wolfram.com/SymmetricMatrix.html
  covariance_matrix = make.positive.definite(covariance_matrix, tol=1e-3) #https://mathworld.wolfram.com/PositiveDefiniteMatrix.html

  lower_bound = c(lower)
  lower_bound = sapply(lower, as.numeric)
  lower_bound = as.vector(lower_bound)

#uncomment next 3 lines for green
  upper_bound = c(upper)
  upper_bound = sapply(upper, as.numeric)
  upper_bound = as.vector(upper_bound)

  #x <- rtmvnorm(n=num_samples, mean=mean_vector, sigma=covariance_matrix, lower = lower_bound)
  #x <- rtmvnorm(n=num_samples, mean=mean_vector, sigma=covariance_matrix,lower = lower_bound[1], upper = lower_bound[0])
  x <- rtmvnorm(n = num_samples, mean=mean_vector, sigma=covariance_matrix, lower=lower_bound ,upper=upper_bound) #for green

  df = data.frame(x)

  return(df)
}

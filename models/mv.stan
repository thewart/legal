data {
  int L;  // lower censoring
  int U;  // upper censoring
  int<lower=0> Nsub;  // number of subjects
  int<lower=0> Nc;  // number of cases
  int<lower=1> Nr;  // number of outcomes/ratings
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of regressors
  real<lower=L, upper=U> R[N];  // ratings
  int<lower=1, upper=Nr> Ri[N];  // rating type
  int<lower=-1, upper=1> cens[N];  // -1 = left censor, 1 = right censor, 0 = none
  matrix[N, P] X;  // design matrix for data
  int<lower=0> S[N];  // subject corresponding to each rating
  int<lower=0> C[N];  // case corresponding to each rating
}

transformed data {
  real M;

  M = (U + L)/2.;
}
parameters {
  // mean and variance across scenarios for each regressor 
  matrix[P, Nr] mu;  
  matrix<lower=0>[P, Nr] eta;  
  cholesky_factor_corr[Nr] L_eta[P];
  matrix<lower=0>[P, Nr] tau;
  real<lower=0> sigma[Nr];  // observation noise
  
  // random effects
  matrix[P, Nr] delta[Nc];  // scenario-specific
  matrix[P, Nr] eps[Nsub];  // subject-specific
  
  
}
transformed parameters {
  matrix[P, Nr] gamma[Nc];  // scenario effects
  matrix[P, Nr] beta[Nsub, Nc];  // individual effects
  
  // draw scenario effects for each group
  // assume population mean effects may be correlated across different rating types
  for (c in 1:Nc) {
    for (p in 1:P) {
      gamma[c, p] = mu[p] + (diag_pre_multiply(eta[p], L_eta[p]) * delta[c, p]')';  // case effects
    }
  }

  // draw individual effects
  // no correlation among subject-specific residuals (eta) given gammas
  for (c in 1:Nc) {
    for (i in 1:Nsub) {
      beta[i, c] = gamma[c] + tau .* eps[i];  // individual effects
    }
  }

}

model {

  //random effects  
  for (i in 1:Nsub) 
    to_vector(eps[i]) ~ normal(0., 1.);  // subject residuals
  for (c in 1:Nc)
    to_vector(delta[c]) ~ normal(0., 1.);  // case residuals
  
  //population parameters
  for (p in 1:P) {
    mu[p] ~ normal(M, M/2);
    eta[p] ~ normal(0, M/4);  // subject variances
    tau[p] ~ normal(0, M/4);  // case variances
    L_eta[p] ~ lkj_corr_cholesky(2.0);
  }
  sigma ~ normal(0, M/4.);
  
  for (i in 1:N) {
    real theta;
    
  // calculate linear predictor
    theta = dot_product(X[i], beta[S[i], C[i], :, Ri[i]]);
      
    if (cens[i] == 0) 
      R[i] ~ normal(theta, sigma[Ri[i]]);
    else if (cens[i] == -1)
      target += normal_lcdf(L | theta, sigma[Ri[i]]);
    else if (cens[i] == 1)
      target += normal_lccdf(U | theta, sigma[Ri[i]]);
  }
}

generated quantities {
  corr_matrix[Nr] Omega[P];
  
  for (p in 1:P) {
    Omega[p] = multiply_lower_tri_self_transpose(L_eta[p]);
  }
}


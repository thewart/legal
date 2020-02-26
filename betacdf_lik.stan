data {
  int L;  // lower censoring
  int U;  // upper censoring
  int D; // distance between consecutive points on scale
  int<lower=0> Nsub;  // number of subjects
  int<lower=0> Nc;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of regressors
  real<lower=L, upper=U> R[N];  // ratings
  int<lower=-1, upper=1> cens[N];  // -1 = left censor, 1 = right censor, 0 = none
  matrix[N, P] X;  // design matrix for data
  int<lower=0> S[N];  // subject corresponding to each rating
  int<lower=0> C[N];  // case corresponding to each rating
  // int<lower=1> K;    //components of mixture transfer function
  // int<lower=1> ns; //number of samples for estimating pointwise predictive liklihood
}

transformed data {
  real M;
  real I;
  real<lower=0,upper=1> Q[N];
  
  M = (U + L)/2.;
  I = D/(2.*(U-L));
  for (i in 1:N)
    Q[i] = (R[i]-L)/(U-L);
}
parameters {
  // mean for each regressor
  vector[P] mu;
  
  // variance across scenarios
  vector<lower=0>[P] eta;
  
  // variance across subjects
  vector<lower=0>[P] tau;
  
  // random effects
  vector[P] delta[Nc];  // scenario-specific
  vector[P] eps[Nsub];  // subject-specific
  
  real<lower=0> inv_scale; //scale of internal transfer function
}

transformed parameters {
  vector[N] theta;
  vector[P] gamma[Nc];  // scenario effects
  vector[P] beta[Nsub, Nc];  // individual effects
  real log_lik[N];
  
  // draw scenario effects for each group
  for (c in 1:Nc)
    gamma[c] = mu + eta .* delta[c];
  // draw individual effects
  for (c in 1:Nc)
    for (i in 1:Nsub) 
      beta[i, c] = gamma[c] + tau .* eps[i];
  // get linear predictor
  for (i in 1:N) {
    
    theta[i] = inv_logit(dot_product(X[i], beta[S[i],C[i]]));
    if (cens[i] == 0)
      log_lik[i] = log_diff_exp(beta_lcdf(Q[i]+I | theta[i]*inv_scale, (1-theta[i])*inv_scale), 
        beta_lcdf(Q[i]-I | theta[i]*inv_scale, (1-theta[i])*inv_scale));
    else if (cens[i] == -1)
      log_lik[i] = beta_lcdf(Q[i]+I | theta[i]*inv_scale, (1-theta[i])*inv_scale);
    else if (cens[i] == 1)
      log_lik[i] = beta_lccdf(Q[i]-I | theta[i]*inv_scale, (1-theta[i])*inv_scale);
  }
}

model {

  for (i in 1:N)
    target += log_lik[i];
    
  mu ~ normal(0, 2.5);
  eta ~ normal(0, 2.5);
  tau ~ normal(0, 2.5);
  
  for (i in 1:Nsub)
    eps[i] ~ normal(0., 1.);
  for (c in 1:Nc)
    delta[c] ~ normal(0., 1.);
  
  inv_scale ~ normal(0, 2.5);
}

data {
  int L;  // lower censoring
  int U;  // upper censoring
  int D; // distance between consecutive points on scale
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of regressors
  real<lower=L, upper=U> Y[N];  // ratings
  int<lower=-1, upper=1> cens[N];  // -1 = left censor, 1 = right censor, 0 = none
  matrix[N, P] X;  // design matrix for main effects
  int<lower=0> Subj[N];  // subject corresponding to each rating
  int<lower=0> Scen[N];  // case corresponding to each rating
  // int<lower=1> K;    //components of mixture transfer fuNscention
}

transformed data {
  real I;
  real<lower=0,upper=1> Q[N];
  
  I = D/(2.*(U-L));
  for (i in 1:N)
    Q[i] = (Y[i]-L)/(U-L);
}

parameters {
  // mean for each regressor
  vector[P] mu;
  
  // variance across scenarios
  vector<lower=0>[P] eta;
  
  // variance across subjects
  vector<lower=0>[P] tau;
  
  // random effects
  vector[P] delta[Nscen];  // scenario-specific
  vector[P] eps[Nsubj];  // subject-specific
  
  real<lower=0> sigma; //scale of internal transfer fuNscention
}

transformed parameters {
  vector[N] theta;
  vector[P] gamma[Nscen];  // scenario effects
  vector[P] beta[Nsubj, Nscen];  // individual effects
  real log_lik[N];
  
  // draw scenario effects for each group
  for (c in 1:Nscen)
    gamma[c] = mu + eta .* delta[c];
  // draw individual effects
  for (c in 1:Nscen)
    for (i in 1:Nsubj) 
      beta[i, c] = gamma[c] + tau .* eps[i];
  // get linear predictor
  for (i in 1:N) {
    
    theta[i] = dot_product(X[i], beta[Subj[i],Scen[i]]);
    if (cens[i] == 0)
      log_lik[i] = log_diff_exp(normal_lcdf(inv_Phi(Q[i]+I)*sigma | theta[i], 1), 
        normal_lcdf(inv_Phi(Q[i]-I)*sigma | theta[i], 1));
    else if (cens[i] == -1)
      log_lik[i] = normal_lcdf(inv_Phi(Q[i]+I)*sigma | theta[i], 1);
    else if (cens[i] == 1)
      log_lik[i] = normal_lccdf(inv_Phi(Q[i]-I)*sigma | theta[i], 1);
  }
}

model {

  for (i in 1:N)
    target += log_lik[i];
    
  mu ~ normal(0, 2.5);
  eta ~ normal(0, 2.5);
  tau ~ normal(0, 2.5);
  
  for (i in 1:Nsubj)
    eps[i] ~ normal(0., 1.);
  for (c in 1:Nscen)
    delta[c] ~ normal(0., 1.);
  
  sigma ~ normal(0, 2.5);
}

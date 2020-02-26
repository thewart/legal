data {
  int L;  // lower censoring
  int U;  // upper censoring
  int D; // distance between consecutive points on scale
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  int<lower=0> P0;  // number of fixed effects
  real<lower=L, upper=U> R[N];  // ratings
  int<lower=-1, upper=1> cens[N];  // -1 = left censor, 1 = right censor, 0 = none
  matrix[N, P] X;  // design matrix for fixed + random effects
  int<lower=0> Subj[N];  // subject corresponding to each rating
  int<lower=0> Scen[N];  // case corresponding to each rating
  // int<lower=1> K;    //components of mixture transfer fuNscention
}

transformed data {
  real I;
  real<lower=0,upper=1> Q[N];
  
  I = D/(2.*(U-L));
  for (i in 1:N)
    Q[i] = (R[i]-L)/(U-L);
}

parameters {
  // mean for each fixed + random eff
  vector[P] beta_mu;
  
  // variance across scenarios
  vector<lower=0>[P] sigma_scen;
  
  // variance across subjects
  vector<lower=0>[P] sigma_subj;
  
  // random effects
  vector[P] beta_scen_raw[Nscen];  // scenario effects
  vector[P] beta_subj_raw[Nsubj];  // subject residual effects
  
  real<lower=0> sigma; //scale of internal transfer function
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  real eta[N]; //linear predictor
  real log_lik[N];

  //random effects
  for (i in 1:Nscen) 
    beta_scen[i] = sigma_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj)
    beta_subj[i] = sigma_subj .* beta_subj_raw[i];
  
  // get linear predictor
  for (i in 1:N) {
    
    eta[i] = X[i]*(beta_mu + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
    if (cens[i] == 0)
      log_lik[i] = log_diff_exp(normal_lcdf(inv_Phi(Q[i]+I)*sigma | eta[i], 1), 
        normal_lcdf(inv_Phi(Q[i]-I)*sigma | eta[i], 1));
    else if (cens[i] == -1)
      log_lik[i] = normal_lcdf(inv_Phi(Q[i]+I)*sigma | eta[i], 1);
    else if (cens[i] == 1)
      log_lik[i] = normal_lccdf(inv_Phi(Q[i]-I)*sigma | eta[i], 1);
  }
}

model {

  for (i in 1:N)
    target += log_lik[i];
    
  beta_mu ~ normal(0, 2.5);
  sigma_scen ~ normal(0, 2.5);
  sigma_subj ~ normal(0, 2.5);
  
  for (i in 1:Nsubj)
    beta_subj_raw[i] ~ normal(0., 1.);
  for (i in 1:Nscen)
    beta_scen_raw[i] ~ normal(0., 1.);
  
  sigma ~ normal(0, 2.5);
}

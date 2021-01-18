data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  matrix[N, P] X;  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int Y[N]; // guilt judgement
}

parameters {
  
  real mu_alpha;
  real<lower=0> sigma_alpha_scen;
  real<lower=0> sigma_alpha_subj;
  
  vector[Nscen] alpha_scen_raw;
  vector[Nsubj] alpha_subj_raw;

  vector[P] mu_beta;
  vector<lower=0>[P] sigma_beta_scen;
  vector<lower=0>[P] sigma_beta_subj;

  vector[P] beta_scen_raw[Nscen];  // scenario effects
  vector[P] beta_subj_raw[Nsubj];  // subject residual effects
  
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  vector[Nscen] alpha_scen = sigma_alpha_scen * alpha_scen_raw;
  vector[Nsubj] alpha_subj = sigma_alpha_subj * alpha_subj_raw;
  vector<lower=0,upper=1>[Nsubj] eps = inv_logit(mu_eps + sigma_eps * eps_raw);
  real log_lik[N];

  //random effects
  for (i in 1:Nscen)
    beta_scen[i] = sigma_beta_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj) 
    beta_subj[i] = sigma_beta_subj .* beta_subj_raw[i];
    
  //linear predictor  
  for (i in 1:N) {
    real eta = mu_alpha + alpha_scen[Scen[i]] + alpha_subj[Subj[i]] + X[i]*(mu_beta + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
    log_lik[i] = log_mix(eps[Subj[i]], bernoulli_lpmf(Y[i] | 0.5), bernoulli_logit_lpmf(Y[i] | eta));
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
      
    mu_alpha ~ normal(0, 2.5);
    sigma_alpha_scen ~ normal(0, 1);
    sigma_alpha_subj ~ normal(0, 1);
    
    mu_beta ~ normal(0, 2.5);
    sigma_beta_scen ~ normal(0, 1);
    sigma_beta_subj ~ normal(0, 1);
    
    mu_eps ~ normal(-2.5, 5);
    sigma_eps ~ normal(5, 5);
    
    alpha_subj_raw ~ normal(0,1);
    alpha_scen_raw ~ normal(0,1);
    eps_raw ~ normal(0, 1);
    for (i in 1:Nsubj)
      beta_subj_raw[i] ~ normal(0, 1);
    for (i in 1:Nscen)
      beta_scen_raw[i] ~ normal(0, 1);
}

data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  matrix[N, P] X;  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=0,upper=1> Y[N]; // guilt judgement
  int<lower=0> P2; // number of interactions
  matrix[N, P2] Z;    // design matrix for pairwise interactions -- must not include intercept!
}

parameters {
  // mean for each fixed + random eff
  vector[P] beta_mu;
  
  // variance across scenarios
  vector<lower=0>[P] sigma_scen;
  
  // residual variances across subjects
  vector<lower=0>[P] sigma_subj;

  // random effects
  vector[P] beta_scen_raw[Nscen];  // scenario effects
  vector[P] beta_subj_raw[Nsubj];  // subject residual effects
  
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
  
  //interactions
  vector[P2] lambda_mu_raw;
  
  real mu_gamma;
  vector[Nsubj] gamma_raw;
  real<lower=0> sigma_gamma;
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  real<lower=0,upper=1> eps[Nsubj];
  real<lower=0> sigma_lambda_subj[Nsubj]; 
  vector[P2] lambda[Nsubj];
  real log_lik[N];

  //random effects
  for (i in 1:Nscen) 
    beta_scen[i] = sigma_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj) {
    beta_subj[i] = sigma_subj .* beta_subj_raw[i];
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
    sigma_lambda_subj[i] = exp(mu_gamma + sigma_gamma * gamma_raw[i]);
    lambda[i] = lambda_mu_raw * sigma_lambda_subj[i];
  }
  //linear predictor  
  for (i in 1:N) {
    real eta = X[i]*(beta_mu + beta_scen[Scen[i]] + beta_subj[Subj[i]]) + Z[i]*lambda[Subj[i]];
    log_lik[i] = log_mix(eps[Subj[i]], bernoulli_lpmf(Y[i] | 0.5), bernoulli_logit_lpmf(Y[i] | eta));
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
    
    beta_mu ~ normal(0, 5);
    sigma_scen ~ normal(0, 1);
    sigma_subj ~ normal(0, 1);
    
    mu_eps ~ normal(0, 5);
    sigma_eps ~ normal(0, 5);
    
    lambda_mu_raw ~ normal(0, 1);
    mu_gamma ~ normal(0, 2.5);
    sigma_gamma ~ normal(0, 2.5);
    
    gamma_raw ~ normal(0, 1);
    eps_raw ~ normal(0, 1);
    for (i in 1:Nsubj) 
      beta_subj_raw[i] ~ normal(0, 1);
    for (i in 1:Nscen)
      beta_scen_raw[i] ~ normal(0, 1);
}

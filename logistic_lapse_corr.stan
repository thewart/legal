data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  vector[P] X[N];  // design matrix for fixed effects
  // int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=2> D;
  int Y[D,N]; // guilt judgement
}

parameters {
  vector[D] alpha_mu;
  vector<lower=0>[D] sigma_alpha;
  vector[D] alpha_raw[Nsubj];
  cholesky_factor_corr[D] L_alpha;
  
  // mean for each fixed + random eff
  matrix[D,P] beta_mu;
  
  // variance across scenarios
  // vector<lower=0>[P] sigma_scen;
  
  // residual variances across subjects
  matrix<lower=0>[D,P] sigma_subj;
  
  vector[D] gamma_mu;
  vector<lower=0>[D] sigma_gamma;
  vector[D] gamma_subj_raw[Nsubj];

  // random effects
  // vector[P] beta_scen_raw[Nscen];  // scenario effects
  matrix[D,P] beta_subj_raw[Nsubj];   // subject residual effects
  vector[N] delta;         //ugh
  
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  // vector[P] beta_scen[Nscen];  // scenario effects
  matrix[D,P] beta_subj[Nsubj];  // individual effects
  vector[D] alpha_subj[Nsubj];
  real<lower=0,upper=1> eps[Nsubj];
  real log_lik[N];
  vector[D] sigma_y[Nsubj];

  //random effects
  // for (i in 1:Nscen) 
  //   beta_scen[i] = sigma_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj) {
    beta_subj[i] = sigma_subj .* beta_subj_raw[i];
    alpha_subj[i] = L_alpha * sigma_alpha .* alpha_raw[i];
    sigma_y[i] = exp(gamma_mu + sigma_gamma .* gamma_subj_raw[i]); 
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
  }
  
  //linear predictor  
  for (i in 1:N) {
    // real eta = X[i]*(beta_mu + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
    vector[D] eta = alpha_mu + alpha_subj[Subj[i]] + (beta_mu + beta_subj[Subj[i]]) * X[i] + sigma_y[Subj[i]] * delta[i];
    log_lik[i] = log_mix(eps[Subj[i]], D*log(0.5), bernoulli_logit_lpmf(Y[:,i] | eta));
  }
}

model {
    
    for (i in 1:N) {
      delta[i] ~ normal(0,1);
      target += log_lik[i];
    }
    
    to_vector(beta_mu) ~ normal(0,10);
    // sigma_scen ~ normal(0, 1);
    to_vector(sigma_subj) ~ normal(0,10);
    
    alpha_mu ~ normal(0,10);
    sigma_alpha ~ normal(0,10);
    L_alpha ~ lkj_corr_cholesky(1);
    
    gamma_mu ~ normal(0,2.5);
    sigma_gamma ~ normal(0,1);

    mu_eps ~ normal(0,5);
    sigma_eps ~ normal(0,5);
    for (i in 1:Nsubj) {
      to_vector(beta_subj_raw[i]) ~ normal(0,1);
      alpha_raw[i] ~ normal(0,1);
      gamma_subj_raw[i] ~ normal(0,1);
      eps_raw[i] ~ normal(0,1);
    }
    
    // to_vector(delta) ~ normal(0,1);
    
    // for (i in 1:Nscen)
    //   beta_scen_raw[i] ~ normal(0., 1.);
}

generated quantities {
  corr_matrix[D] Rho_alpha = L_alpha*L_alpha';
  // corr_matrix[D] Rho_y = L_y*L_y';
}


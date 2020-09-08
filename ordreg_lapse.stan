data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  matrix[N, P] X;  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int Y[N]; // guilt judgement
  int ps;
}

parameters {
  // mean for each fixed + random eff
  vector[P] mu_beta;
  
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
  
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[Nsubj] alpha_raw;
  
  real mu_theta;
  real<lower=0> sigma_theta;
  vector[Nsubj] theta_raw;
  
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  vector<lower=0,upper=1>[Nsubj] eps = inv_logit(mu_eps + sigma_eps * eps_raw);
  real log_lik[N];
  vector[Nsubj] alpha = mu_alpha + sigma_alpha * alpha_raw;
  vector[Nsubj] theta = exp(mu_theta + sigma_theta * theta_raw);
  vector[2] cp[Nsubj];
  
  //random effects
  for (i in 1:Nscen) 
    beta_scen[i] = sigma_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj) {
    beta_subj[i] = sigma_subj .* beta_subj_raw[i];
    cp[i][1] = alpha[i];
    cp[i][2] = alpha[i] + theta[i];
  }
  //linear predictor  
  for (i in 1:N) {
    if (Y[i] > 0) {
      real eta = X[i]*(mu_beta + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
      log_lik[i] = log_mix(eps[Subj[i]], log(0.5)*2,
                                       ordered_logistic_lpmf(Y[i] | eta, cp[Subj[i]]));
    } else {
      log_lik[i] = eps[Subj[i]] + log(0.5)*2;
    }
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
    
    mu_beta ~ normal(0, 2.5);
    sigma_scen ~ normal(0, 1);
    sigma_subj ~ normal(0, 1);
    
    mu_alpha ~ normal(0,2.5);
    sigma_alpha ~ normal(0,1);
    mu_theta ~ normal(0,2.5);
    sigma_theta ~ normal(0,1);
    
    mu_eps ~ normal(0, 5);
    sigma_eps ~ normal(0, 5);
    
    for (i in 1:Nsubj)
      beta_subj_raw[i] ~ normal(0,1);
      
    for (i in 1:Nscen)
      beta_scen_raw[i] ~ normal(0, 1);
    
    alpha_raw ~ normal(0,1);
    theta_raw ~ normal(0,1);
    eps_raw ~ normal(0,1);
}

generated quantities {
  real mu_legalgap;
  real sigma_legalgap;
  real rho_alpha_theta = (mean(alpha .* theta) - mean(alpha)*mean(theta))/(sd(alpha) * sd(theta));

  real mu_lapserate;
  real sigma_lapserate;
  
  {
    vector[ps] x;
    vector[ps] y;
    for (i in 1:ps) {
      x[i] = exp(mu_theta + sigma_theta * normal_rng(0,1));
      y[i] = inv_logit(mu_eps + sigma_eps * normal_rng(0,1));
    }
    mu_legalgap = mean(x);
    sigma_legalgap = sd(x);
    mu_lapserate = mean(y);
    sigma_lapserate = sd(y);
  }
}

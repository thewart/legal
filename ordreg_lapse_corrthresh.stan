data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  matrix[N, P] X;  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int Y[N]; // guilt judgement
  int<lower=2> Nthresh;
  int ps;
}

parameters {
  // mean for each fixed + random eff
  vector[P] mu_beta;
  vector<lower=0>[P] sigma_beta_subj;
  vector<lower=0>[P] sigma_beta_scen;
  
  real mu_eps;
  real<lower=0> sigma_eps;
  
  real mu_alpha;
  real<lower=0> sigma_alpha_subj;
  real<lower=0> sigma_alpha_scen;
  
  real mu_theta[Nthresh-1];
  real<lower=0> sigma_theta_subj[Nthresh-1];
  real<lower=0> sigma_theta_scen[Nthresh-1];
  
  cholesky_factor_corr[Nthresh] L_thresh_subj;
  cholesky_factor_corr[Nthresh] L_thresh_scen;
  
  // random effects
  vector[P] beta_scen_raw[Nscen];  // scenario effects
  vector[P] beta_subj_raw[Nsubj];  // subject residual effects
  vector[Nsubj] alpha_subj_raw;
  vector[Nscen] alpha_scen_raw;
  vector[Nsubj] theta_subj_raw;
  vector[Nscen] theta_scen_raw;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  vector<lower=0,upper=1>[Nsubj] eps = inv_logit(mu_eps + sigma_eps * eps_raw);
  real log_lik[N];
  vector[Nthresh] cp_subj[Nsubj];
  vector[Nthresh] cp_scen[Nscen];
  
  //random effects
  {
    for (i in 1:Nscen) {
      cp_scen[i][1] = sigma_alpha_scen * alpha_scen_raw[i];
      cp_scen[i][2] = sigma_theta_scen * theta_scen_raw[i];
      cp_scen[i] = L_thresh_scen * cp_scen[i];
      beta_scen[i] = sigma_beta_scen .* beta_scen_raw[i];
    }
    
    for (i in 1:Nsubj) {
      cp_subj[i][1] = sigma_alpha_subj * alpha_subj_raw[i];
      cp_subj[i][2] = sigma_theta_subj * theta_subj_raw[i];
      cp_subj[i] = L_thresh_scen * cp_subj[i];
      beta_subj[i] = sigma_beta_subj .* beta_subj_raw[i];
    }
  }
  
  for (i in 1:N) {
    if (Y[i] > 0) {
      vector[2] cp;
      real eta = X[i]*(mu_beta + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
      cp[1] = mu_alpha + cp_subj[Subj[i]][1] + cp_scen[Scen[i]][1];
      cp[2] = cp[1] + exp(mu_theta + cp_subj[Subj[i]][2] + cp_scen[Scen[i]][2]);
      log_lik[i] = log_mix(eps[Subj[i]], log(0.5)*2, ordered_logistic_lpmf(Y[i] | eta, cp));
    } else {
      log_lik[i] = eps[Subj[i]] + log(0.5)*2;
    }
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
    
    mu_beta ~ normal(0, 2.5);
    sigma_beta_scen ~ normal(0, 1);
    sigma_beta_subj ~ normal(0, 1);
    
    mu_alpha ~ normal(0,2.5);
    sigma_alpha_scen ~ normal(0,1);
    sigma_alpha_subj ~ normal(0,1);
    mu_theta ~ normal(0,2.5);
    sigma_theta_scen ~ normal(0,1);
    sigma_theta_subj ~ normal(0,1);
    
    mu_eps ~ normal(-2.5, 5);
    sigma_eps ~ normal(5, 5);
    
    for (i in 1:Nsubj)
      beta_subj_raw[i] ~ normal(0,1);
      
    for (i in 1:Nscen)
      beta_scen_raw[i] ~ normal(0, 1);
    
    alpha_subj_raw ~ normal(0,1);
    alpha_scen_raw ~ normal(0,1);
    theta_scen_raw ~ normal(0,1);
    theta_subj_raw ~ normal(0,1);
    eps_raw ~ normal(0,1);
}

generated quantities {
  real mu_legalgap;
  real sigma_legalgap;
  // real rho_thresh = (mean(alpha .* theta) - mean(alpha)*mean(theta))/(sd(alpha) * sd(theta));
  real rho_thresh;
  real mu_lapserate;
  real sigma_lapserate;
  
  {
    vector[ps] x;
    vector[ps] y;
    vector[ps] z;
    for (i in 1:ps) {
      vector[2] cps_scen;
      vector[2] cps_subj;
      
      cps_scen[1] = sigma_alpha_scen * normal_rng(0,1);
      cps_scen[2] = sigma_theta_scen * normal_rng(0,1);
      cps_scen = L_thresh_scen * cps_scen;
      cps_subj[1] = sigma_alpha_subj * normal_rng(0,1);
      cps_subj[2] = sigma_theta_subj * normal_rng(0,1);
      cps_subj = L_thresh_subj * cps_subj;
      
      z[i] = mu_alpha + cps_scen[1] + cps_subj[1];
      x[i] = exp(mu_theta + cps_scen[2] + cps_subj[2]);
      y[i] = inv_logit(mu_eps + sigma_eps * normal_rng(0,1));
    }
    mu_legalgap = mean(x);
    sigma_legalgap = sd(x);
    rho_thresh = (mean(x .* z) - mean(x) * mean(z))/(sd(x) * sd(z));
    mu_lapserate = mean(y);
    sigma_lapserate = sd(y);
  }
}

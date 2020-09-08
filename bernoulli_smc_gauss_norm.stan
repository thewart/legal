functions {
  real scalemixcdf(real x, vector w, vector loc, vector scale) {
    real y = 0;
    int K = num_elements(w);
    for (i in 1:K)
      y += w[i]*normal_cdf(x, loc[i], scale[i]);
    return y;
  }

  real bernoulli_smc_lpmf(int y, real x, vector w, vector loc, vector scale) {
    return bernoulli_lpmf(y | scalemixcdf(x, w, loc, scale));
  }
}

data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  matrix[N, P] X;  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=0,upper=1> Y[N]; // guilt judgement
  int<lower=1> K; //number of mixture components in transfer function
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
  
  simplex[K] w; //weights on transfunc mixture components
  simplex[K] scale_raw;
  vector<lower=0>[K-1] loc_dist;     //component location distances
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  real<lower=0,upper=1> eps[Nsubj];
  vector[K] loc;
  vector[K] scale;
  real log_lik[N];
  
  // scale = cumulative_sum(scale_diffs);
  {
    real presd;
    loc = append_row(0, cumulative_sum(loc_dist));
    loc = loc - dot_product(w, loc);
    
    presd = sqrt(dot_product(w, loc .* loc + scale_raw .* scale_raw));
    loc = loc/presd;
    scale = scale_raw/presd;
  }
  
  //random effects
  for (i in 1:Nscen) 
    beta_scen[i] = sigma_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj) {
    beta_subj[i] = sigma_subj .* beta_subj_raw[i];
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
  }
  //linear predictor  
  for (i in 1:N) {
    real eta = X[i]*(beta_mu + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
    log_lik[i] = log_mix(eps[Subj[i]], bernoulli_lpmf(Y[i] | 0.5),
                                       bernoulli_smc_lpmf(Y[i] | eta,w,loc,scale));
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
    
    beta_mu ~ normal(0, 2.5);
    sigma_scen ~ normal(0, 1);
    sigma_subj ~ normal(0, 1);
    
    mu_eps ~ normal(0, 5);
    sigma_eps ~ normal(0, 10);
    
    w ~ dirichlet(rep_vector(1,K));
    scale_raw ~ dirichlet(rep_vector(1,K));
    // scale ~ normal(0,1);
    loc_dist ~ normal(0,1);
    
    for (i in 1:Nsubj) {
      beta_subj_raw[i] ~ normal(0., 1.);
      eps_raw[i] ~ normal(0., 1.);
    }
    for (i in 1:Nscen)
      beta_scen_raw[i] ~ normal(0., 1.);
}

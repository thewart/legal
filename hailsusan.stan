data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> Pe;  // number of exculpatory evidence types
  int<lower=0> Pi;  // number of inculpatory evidence types
  matrix[N, Pe] Xe;  // design matrix for exculpatory evidence
  matrix[N, Pi] Xi;  //design matrix for inculpatory evidence
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=0,upper=1> Y[N]; // guilt judgement
}

parameters {
  
  //population net evidence weights
  vector<lower=0>[Pe] mu_beta_exc;
  vector<lower=0>[Pi] mu_beta_inc;
  real mu_beta_ixe;
  vector<lower=0>[Pe] sigma_exc;
  vector<lower=0>[Pi] sigma_inc;
  real<lower=0> sigma_ixe;

  //subject-specific relative evidence weights
  //simplex[Pe] w_exc[Nsubj];
  //simplex[Pi] w_inc[Nsubj];
  //subject-specific net evidence weights
  vector[Pe] beta_exc_raw[Nsubj];
  vector[Pi] beta_inc_raw[Nsubj];
  real beta_ixe_raw[Nsubj];

  real mu_beta_0;
  real<lower=0> sigma_0;
  real beta_0_raw[Nsubj];
  
  //lapse mixture
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  real<lower=0,upper=1> eps[Nsubj];
  real beta_0[Nsubj];
  vector[Pe] beta_exc[Nsubj];
  vector[Pi] beta_inc[Nsubj];
  real beta_ixe[Nsubj];
  real log_lik[N];
  
  for (i in 1:Nsubj) {
    beta_0[i] = mu_beta_0 + sigma_0 * beta_0_raw[i];
    beta_exc[i] = mu_beta_exc + sigma_exc .* beta_exc_raw[i];
    beta_inc[i] = mu_beta_inc + sigma_inc .* beta_inc_raw[i];
    beta_ixe[i] = mu_beta_ixe + sigma_ixe * beta_ixe_raw[i];
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
  }


  //linear predictor  
  for (i in 1:N) {
    real eta_inc_norm = Xi[Subj[i]]*(beta_inc[Subj[i]]/sum(beta_inc[Subj[i]]));
    real eta_exc_norm = Xe[Subj[i]]*(beta_exc[Subj[i]]/sum(beta_exc[Subj[i]]));
    real eta = beta_0[Subj[i]] - Xe[i]*mu_beta_exc + Xi[i]*mu_beta_inc +
      beta_ixe[Subj[i]]*eta_inc_norm*eta_exc_norm;
    log_lik[i] = log_mix(eps[Subj[i]], bernoulli_lpmf(Y[i] | 0.5),
                                       bernoulli_logit_lpmf(Y[i] | eta));
  }
}

model {
    for (i in 1:N)
      target += log_lik[i];
    
    mu_beta_0 ~ normal(0, 2.5);
    mu_beta_exc ~ normal(0, 2.5);
    mu_beta_inc ~ normal(0, 2.5);
    mu_beta_ixe ~ normal(0, 2.5);
    sigma_exc ~ normal(0, 1);
    sigma_inc ~ normal(0, 1);
    sigma_ixe ~ normal(0, 1);
    sigma_0 ~ normal(0, 1);
    
    // mu_w_exc ~ dirichlet(rep_vector(1,Pe));
    // mu_w_inc ~ dirichlet(rep_vector(1,Pi));
    // logalpha_exc ~ normal(0,10);
    // logalpha_inc ~ normal(0,10);
    
    mu_eps ~ normal(0, 5);
    sigma_eps ~ normal(0, 10);
    for (i in 1:Nsubj) {
      beta_0_raw[i] ~ normal(0, 1);
      eps_raw ~ normal(0, 1);
      beta_exc_raw[i] ~ normal(0, 1);
      beta_inc_raw[i] ~ normal(0, 1);
      beta_ixe_raw[i] ~ normal(0, 1);
      // w_exc[i] ~ dirichlet(exp(logalpha_exc)*mu_w_exc);
      // w_inc[i] ~ dirichlet(exp(logalpha_inc)*mu_w_inc);
    }
}

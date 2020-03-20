data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> Pe;  // number of exculpatory evidence types
  int<lower=0> Pi;  // number of inculpatory evidence types
  int<lower=0> Pa;  // number of ambiguous evidence types
  matrix[N, Pe] Xe;  // design matrix for exculpatory evidence
  matrix[N, Pi] Xi;  //design matrix for inculpatory evidence
  matrix[N, Pa] Xa;
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=0,upper=1> Y[N]; // guilt judgement
}

parameters {
  
  //population mean relative evidence weights
  simplex[Pe] mu_w_exc;
  simplex[Pi] mu_w_inc;
  vector[Pa] mu_beta_amb;
  vector[Pa] mu_beta_axi;
  vector[Pa] mu_beta_axe;
  //population concentration of relative evidence weights
  //real<lower=0> logalpha_exc;
  //real<lower=0> logalpha_inc;
  
  //population net evidence weights
  real mu_beta_exc;
  real mu_beta_inc;
  //real mu_beta_ixe;
  real<lower=0> sigma_exc;
  real<lower=0> sigma_inc;
  //real<lower=0> sigma_ixe;
  vector<lower=0>[Pa] sigma_amb;
  vector<lower=0>[Pa] sigma_axe;
  vector<lower=0>[Pa] sigma_axi;
  vector<lower=0>[Pa] sigma_scen_amb;
  vector<lower=0>[Pa] sigma_scen_axe;
  vector<lower=0>[Pa] sigma_scen_axi;

  //subject-specific relative evidence weights
  //simplex[Pe] w_exc[Nsubj];
  //simplex[Pi] w_inc[Nsubj];
  //subject-specific net evidence weights
  real beta_exc_raw[Nsubj];
  real beta_inc_raw[Nsubj];
  //real beta_ixe_raw[Nsubj];
  vector[Pa] beta_amb_raw[Nsubj];
  vector[Pa] beta_axi_raw[Nsubj];
  vector[Pa] beta_axe_raw[Nsubj];
  
  real mu_beta_0;
  real<lower=0> sigma_0_subj;
  real<lower=0> sigma_0_scen;
  real beta_0_subj_raw[Nsubj];
  real beta_0_scen_raw[Nscen];
  
  //lapse mixture
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  real<lower=0,upper=1> eps[Nsubj];
  real beta_exc[Nsubj];
  real beta_inc[Nsubj];
  //real beta_0[Nscen];
  //real beta_ixe[Nsubj];
  vector[Pa] beta_amb[Nsubj];
  vector[Pa] beta_axe[Nsubj];
  vector[Pa] beta_axi[Nsubj];
  real log_lik[N];
  
  for (i in 1:Nsubj) {
    //beta_0[i] = sigma_0 * beta_0_raw[i];
    beta_exc[i] = mu_beta_exc + sigma_exc * beta_exc_raw[i];
    beta_inc[i] = mu_beta_inc + sigma_inc * beta_inc_raw[i];
    //beta_ixe[i] = mu_beta_ixe + sigma_ixe * beta_ixe_raw[i];
    beta_amb[i] = mu_beta_amb + sigma_amb .* beta_amb_raw[i];
    beta_axe[i] = mu_beta_axe + sigma_axe .* beta_axe_raw[i];
    beta_axi[i] = mu_beta_axi + sigma_axi .* beta_axi_raw[i];
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
  }

  //linear predictor  
  for (i in 1:N) {
    real eta_exc = Xe[i]*mu_w_exc;
    real eta_inc = Xi[i]*mu_w_inc;
    real eta = mu_beta_0 + sigma_0_subj*beta_0_subj_raw[Subj[i]] + sigma_0_scen*beta_0_scen_raw[Scen[i]] +
      beta_exc[Subj[i]]*eta_exc + beta_inc[Subj[i]]*eta_inc + 
      //beta_ixe[Subj[i]]*(Xe[i]*mu_w_exc * Xi[i]*mu_w_inc);
      Xa[i]*beta_amb[Subj[i]] + eta_inc*Xa[i]*beta_axi[Subj[i]] + eta_exc*Xa[i]*beta_axe[Subj[i]];
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
    mu_beta_amb ~ normal(0, 2.5);
    mu_beta_axi ~ normal(0, 2.5);
    mu_beta_axe ~ normal(0, 2.5);
    //mu_beta_ixe ~ normal(0, 2.5);
    sigma_exc ~ normal(0, 1);
    sigma_inc ~ normal(0, 1);
    //sigma_ixe ~ normal(0, 1);
    sigma_amb ~ normal(0, 1);
    sigma_axe ~ normal(0, 1);
    sigma_axi ~ normal(0, 1);
    sigma_0_subj ~ normal(0, 1);
    sigma_0_scen ~ normal(0, 1);
    
    mu_w_exc ~ dirichlet(rep_vector(1,Pe));
    mu_w_inc ~ dirichlet(rep_vector(1,Pi));
    //logalpha_exc ~ normal(0,10);
    //logalpha_inc ~ normal(0,10);
    
    mu_eps ~ normal(0, 5);
    sigma_eps ~ normal(0, 10);
    for (i in 1:Nsubj) {
      beta_0_subj_raw[i] ~ normal(0, 1);
      eps_raw ~ normal(0, 1);
      beta_exc_raw[i] ~ normal(0, 1);
      beta_inc_raw[i] ~ normal(0, 1);
      beta_amb_raw[i] ~ normal(0, 1);
      beta_axe_raw[i] ~ normal(0, 1);
      beta_axi_raw[i] ~ normal(0, 1);
      //beta_ixe_raw[i] ~ normal(0, 1);
      //w_exc[i] ~ dirichlet(exp(logalpha_exc)*mu_w_exc);
      //w_inc[i] ~ dirichlet(exp(logalpha_inc)*mu_w_inc);
    }
    for (i in 1:Nscen)
      beta_0_scen_raw[i] ~ normal(0, 1);
}

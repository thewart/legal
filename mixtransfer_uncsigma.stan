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
  real mu_alpha;
  real<lower=0> sigma_alpha_scen;
  real<lower=0> sigma_alpha_subj;
  
  vector[Nscen] alpha_scen_raw;
  vector[Nsubj] alpha_subj_raw;

  // mean for each regressor
  vector[P] mu_beta;
  
  // variance across scenarios
  vector<lower=0>[P] sigma_beta_scen;
  
  // variance across subjects
  vector<lower=0>[P] sigma_beta_subj;
  
  // random effects
  vector[P] beta_scen_raw[Nscen];  // scenario-specific
  vector[P] beta_subj_raw[Nsubj];  // subject-specific
  
  real<lower=0> sigma; //scale of internal transfer fuNscention
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  vector[Nscen] alpha_scen = sigma_alpha_scen * alpha_scen_raw;
  vector[Nsubj] alpha_subj = sigma_alpha_subj * alpha_subj_raw;
  real eta[N]; //linear predictor
  real log_lik[N];

  //random effects
  for (i in 1:Nscen) 
    beta_scen[i] = sigma_beta_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj)
    beta_subj[i] = sigma_beta_subj .* beta_subj_raw[i];
  // get linear predictor
  for (i in 1:N) {
    
    eta[i] = mu_alpha + alpha_scen[Scen[i]] + alpha_subj[Subj[i]] + X[i]*(mu_beta + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
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

  for (i in 1:N) target += log_lik[i];
  
  mu_alpha ~ normal(0, 2.5);
  sigma_alpha_scen ~ normal(0, 1);
  sigma_alpha_subj ~ normal(0, 1);
  
  alpha_subj_raw ~ normal(0,1);
  alpha_scen_raw ~ normal(0,1);
    
  mu_beta ~ normal(0, 2.5);
  sigma_beta_scen ~ normal(0, 2.5);
  sigma_beta_subj ~ normal(0, 2.5);
  
  for (i in 1:Nsubj)
    beta_subj_raw[i] ~ normal(0., 1.);
  for (c in 1:Nscen)
    beta_scen_raw[c] ~ normal(0., 1.);
  
  sigma ~ normal(0, 2.5);
}

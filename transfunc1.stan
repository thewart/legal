data {
  int L;  // lower censoring
  int U;  // upper censoring
  int D; // distance between consecutive points on scale
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  // int<lower=0> P0;  // number of fixed effects
  real<lower=L, upper=U> Y[N];  // ratings
  // int<lower=-1, upper=1> cens[N];  // -1 = left censor, 1 = right censor, 0 = none
  matrix[N, P] X;  // design matrix for fixed + random effects
  int<lower=0> Subj[N];  // subject corresponding to each rating
  int<lower=0> Scen[N];  // case corresponding to each rating
  // int<lower=1> K;    //components of mixture transfer function
  real<lower=1> margin;
}

transformed data {
  real I;
  real<lower=0,upper=1> Q[N];
  int<lower=-1,upper=1> cens[N];
  real QU;
  real QL;
  
  for (i in 1:N) {
    if (Y[i] > (U-margin*D)) {
      cens[i] = 1;
    } else if (Y[i] < (L+margin*D)) {
      cens[i] = -1;
    } else {
      cens[i] = 0;
    }
  }
  
  I = D/(2.*(U-L));
  QU = (U-margin*D+D-L)/(U-L);
  QL = (margin*D-D)/(U-L);
  for (i in 1:N)
    Q[i] = (Y[i]-L)/(U-L);
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
  
  real<lower=0> sigma; //scale of internal transfer function
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
    eta[i] = X[i]*(mu_beta + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
    if (cens[i] == 0)
      log_lik[i] = log_diff_exp(normal_lcdf(inv_Phi(Q[i]+I)*sigma | eta[i], 1), 
        normal_lcdf(inv_Phi(Q[i]-I)*sigma | eta[i], 1));
    else if (cens[i] == -1)
      log_lik[i] = normal_lcdf(inv_Phi(QL+I)*sigma | eta[i], 1);
    else if (cens[i] == 1)
      log_lik[i] = normal_lccdf(inv_Phi(QU-I)*sigma | eta[i], 1);
      
    print(log_lik[i],",",Q[i],",",inv_Phi(Q[i]),",",eta[i],",",sigma);
  }
}

model {
  
  for (i in 1:N) target += log_lik[i];
  
  mu_alpha ~ normal(0, 2.5);
  sigma_alpha_scen ~ normal(0, 1);
  sigma_alpha_subj ~ normal(0, 1);
  
  mu_beta ~ normal(0, 2.5);
  sigma_beta_scen ~ normal(0, 1);
  sigma_beta_subj ~ normal(0, 1);
  
  alpha_subj_raw ~ normal(0,1);
  alpha_scen_raw ~ normal(0,1);
  for (i in 1:Nsubj)
    beta_subj_raw[i] ~ normal(0., 1.);
  for (i in 1:Nscen)
    beta_scen_raw[i] ~ normal(0., 1.);
  
  sigma ~ normal(0, 2.5);
}

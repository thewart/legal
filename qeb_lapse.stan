functions {
  vector interact(vector y) {
    int D = num_elements(y);
    vector[D*(D-1)/2] x;
    int k=1;
    
    for (i in 1:(D-1)) {
      for (j in (i+1):D) {
        x[k] = y[i]*y[j];
        k = k + 1;
      }
    }
    return x;
  }
  
  matrix create_Yset(int D, int D2) {
    matrix[D,D2] Yset;
    for (i in 1:D2) {
      int ind = i-1;
      for (j in 1:D) {
        Yset[j,i] = fmod(ind,2);
        ind = ind/2;
      }
    }
    return Yset;
  }
}

data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  vector[P] X[N];  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=2> D; //data dimensions
  int D2;  //ugh why
  vector[D] Y[N]; // guilt judgement
}

transformed data {
  matrix[D,D2] Yset;
  int Di = D*(D-1)/2;
  matrix[Di,D2] YYset;
  vector[Di] YY[N];
  // matrix[D,N] Yc = 2*Y-1;
  
  Yset = create_Yset(D,D2);
  for (i in 1:D2) YYset[:,i] = interact(Yset[:,i]);
  for (i in 1:N) YY[i] = interact(Y[i]); 
}

parameters {
  
  vector[D] mu_alpha;
  vector<lower=0>[D] sigma_alpha_subj;
  vector<lower=0>[D] sigma_alpha_scen;
  vector[D] alpha_subj_raw[Nsubj];
  vector[D] alpha_scen_raw[Nscen];
  
  // mean for each fixed + random eff
  matrix[D,P] mu_beta;
  matrix<lower=0>[D,P] sigma_beta_scen;
  matrix<lower=0>[D,P] sigma_beta_subj;
  
  // cholesky_factor_corr[D] L_beta_subj;
  // cholesky_factor_corr[D] L_beta_scen;

  // random effects
  matrix[D,P] beta_scen_raw[Nscen];  // scenario effects
  matrix[D,P] beta_subj_raw[Nsubj];  // subject residual effects
  
  vector[Di] mu_f2;
  vector<lower=0>[Di] sigma_f2_subj;
  vector<lower=0>[Di] sigma_f2_scen;

  matrix[Di, Nsubj] f2_subj_raw;
  matrix[Di, Nscen] f2_scen_raw;
  
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  vector[D] alpha_subj[Nsubj];
  vector[D] alpha_scen[Nscen];
  matrix[D,P] beta_scen[Nscen];  // scenario effects
  matrix[D,P] beta_subj[Nsubj];  // individual effects
  matrix[Di,Nscen] f2_scen = diag_pre_multiply(sigma_f2_scen,f2_scen_raw);
  matrix[Di,Nsubj] f2_subj = diag_pre_multiply(sigma_f2_subj, f2_subj_raw);
  real<lower=0,upper=1> eps[Nsubj];
  real log_lik[N];
  
  //random effects
  for (i in 1:Nscen) {
    alpha_scen[i] = sigma_alpha_scen .* alpha_scen_raw[i];
    beta_scen[i] = sigma_beta_scen .* beta_scen_raw[i];
  }
  for (i in 1:Nsubj) {
    alpha_subj[i] = sigma_alpha_subj .* alpha_subj_raw[i];
    beta_subj[i] = sigma_beta_subj .* beta_subj_raw[i];
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
  }
  
  //linear predictor  
  for (i in 1:N) {
    vector[D] f1 = mu_alpha + alpha_scen[Scen[i]] + alpha_subj[Subj[i]] + (mu_beta + beta_scen[Scen[i]] + beta_subj[Subj[i]]) * X[i];
    vector[Di] f2 = mu_f2 + f2_scen[:,Scen[i]] + f2_subj[:,Subj[i]];
    real logZ = log_sum_exp(f1' * Yset + f2' * YYset);
    log_lik[i] = log_mix(eps[Subj[i]], log(0.5)*D, f1' * Y[i] + f2'*YY[i] - logZ);
    // log_lik[i] = f1' * Y[i] + f2'*YY[i] - logZ;
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
  
    to_vector(mu_beta) ~ normal(0,2.5);
    to_vector(sigma_beta_scen) ~ normal(0,1);
    to_vector(sigma_beta_subj) ~ normal(0,1);
    
    to_vector(mu_alpha) ~ normal(0,5);
    to_vector(sigma_alpha_scen) ~ normal(0,1);
    to_vector(sigma_alpha_subj) ~ normal(0,1);
    
    // L_beta_subj ~ lkj_corr_cholesky(1);
    // L_beta_scen ~ lkj_corr_cholesky(1);
    
    to_vector(mu_f2) ~ normal(0,5);
    to_vector(sigma_f2_subj) ~ normal(0,1);
    to_vector(sigma_f2_scen) ~ normal(0,1);
    
    mu_eps ~ normal(-2.5, 5);
    sigma_eps ~ normal(5, 5);
    
    for (i in 1:Nsubj) {
      alpha_subj_raw[i] ~ normal(0,1);
      to_vector(beta_subj_raw[i]) ~ normal(0,1);
      eps_raw[i] ~ normal(0,1);
    }
    for (i in 1:Nscen) {
      alpha_scen_raw[i] ~ normal(0,1);
      to_vector(beta_scen_raw[i]) ~ normal(0,1);
    }
      
    to_vector(f2_subj_raw) ~ normal(0,1);
    to_vector(f2_scen_raw) ~ normal(0,1);
}

// generated quantities {
  // corr_matrix[D] Rho_beta_subj = L_beta_subj * L_beta_subj';
  // corr_matrix[D] Rho_beta_scen = L_beta_scen * L_beta_scen';
// }

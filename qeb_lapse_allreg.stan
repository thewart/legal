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
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=2> D; //data dimensions
  int D2;  //ugh why
  matrix[D,N] Y; // guilt judgement
}

transformed data {
  matrix[D,D2] Yset;
  int Di = D*(D-1)/2;
  int Dt = Di+D;
  matrix[Di,D2] YYset;
  matrix[Di,N] YY;
  matrix[Dt,N] Yt;
  matrix[Dt,D2] Ytset;
  // matrix[D,N] Yc = 2*Y-1;
  
  Yset = create_Yset(D,D2);
  for (i in 1:D2) YYset[:,i] = interact(Yset[:,i]);
  for (i in 1:N) YY[:,i] = interact(Y[:,i]); 
  
  Yt = append_row(Y,YY);
  Ytset = append_row(Yset,YYset);
}

parameters {
  
  vector[Dt] mu_alpha;
  vector<lower=0>[Dt] sigma_alpha;
  cholesky_factor_corr[Dt] L_alpha;
  
  matrix[Dt,P] mu_beta;
  matrix<lower=0>[Dt,P] sigma_beta;
  cholesky_factor_corr[Dt] L_beta;

  // random effects
  matrix[Dt,P] beta_raw[Nsubj];  // subject residual effects
  vector[Dt] alpha_raw[Nsubj];
  
  real mu_eps;
  real<lower=0> sigma_eps;
  vector[Nsubj] eps_raw;  // probability of randomly responding
}

transformed parameters {
  matrix[Dt,P] beta_subj[Nsubj];  // individual effects
  vector[Dt] alpha[Nsubj];
  real<lower=0,upper=1> eps[Nsubj];
  real log_lik[N];
  
  //random effects
  for (i in 1:Nsubj) {
    alpha[i] = L_alpha * sigma_alpha .* alpha_raw[i];
    beta_subj[i] = L_beta * sigma_beta .* beta_raw[i];
    eps[i] = inv_logit(mu_eps + sigma_eps * eps_raw[i]);
  }
  
  //linear predictor  
  for (i in 1:N) {
    vector[Dt] theta = mu_alpha + alpha[Subj[i]] + (mu_beta + beta_subj[Subj[i]]) * X[i];
    real logZ = log_sum_exp(theta'*Ytset);
    log_lik[i] = log_mix(eps[Subj[i]], log(0.5)*D, theta'*Yt[:,i] - logZ);
  }
}

model {
    
    for (i in 1:N)
      target += log_lik[i];
    
    mu_alpha ~ normal(0,2.5);
    sigma_alpha ~ normal(0, 1);
    L_alpha ~ lkj_corr_cholesky(1);
    
    to_vector(mu_beta) ~ normal(0,2.5);
    to_vector(sigma_beta) ~ normal(0,1);
    L_beta ~ lkj_corr_cholesky(1);
    
    mu_eps ~ normal(0,5);
    sigma_eps ~ normal(0,5);
    for (i in 1:Nsubj) {
      to_vector(beta_raw[i]) ~ normal(0,1);
      eps_raw[i] ~ normal(0,1);
      alpha_raw[i] ~ normal(0,1);
    }
}

generated quantities {
  corr_matrix[Dt] Rho_beta = L_beta*L_beta';
  corr_matrix[Dt] Rho_alpha = L_alpha*L_alpha';
}

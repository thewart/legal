data {
  int<lower=0> Nsubj;  // number of subjects
  int<lower=0> Nscen;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of fixed + random effect regressors
  matrix[N, P] X;  // design matrix for fixed effects
  int<lower=0> Scen[N];  // subject corresponding to each rating
  int<lower=0> Subj[N];  // case corresponding to each rating
  int<lower=0,upper=1> Y[N]; // guilt judgement
  int<lower=1> K; //number of factors
}

transformed data {
  int<lower=1> nl;
  nl = (P-1)*K - choose(K,2);
}

parameters {
  // mean for each fixed + random eff
  vector[P] beta_mu;
  
  // variance across scenarios
  vector<lower=0>[P] sigma_scen;
  
  // residual variances across subjects
  vector<lower=0>[P] sigma_subj_resid;
  // factor loadings
  vector[nl] lambda_lower;
  vector<lower=0>[K] lambda_diag;
  // loading variance 
  real<lower=0> sigma_lambda;
  
  // random effects
  vector[P] beta_scen_raw[Nscen];  // scenario effects
  vector[P] beta_subj_raw[Nsubj];  // subject residual effects
}

transformed parameters {
  vector[P] beta_scen[Nscen];  // scenario effects
  vector[P] beta_subj[Nsubj];  // individual effects
  real eta[N]; //linear predictor
  cholesky_factor_cov[P,K] Lambda; //factor loadings matrix
  cholesky_factor_cov[P] L;
  
  //construct loading matrix
  {
    int i=1;
    for (m in 1:P) {
      for (n in 1:K) {
        if (m == n) {
          Lambda[m, n] = lambda_diag[m];
        } else if (m > n) {
          Lambda[m, n] = lambda_lower[i];
          i += 1;
        } else if (m < n) {
          Lambda[m, n] = 0;
        }
      }
    }
  }
  
  //final covariance matrix
  L = cholesky_decompose(tcrossprod(Lambda) + diag_matrix(sigma_subj_resid));
  
  //random effects
  for (i in 1:Nscen) 
    beta_scen[i] = sigma_scen .* beta_scen_raw[i];
  for (i in 1:Nsubj)
    beta_subj[i] = L*beta_subj_raw[i];
  
  //linear predictor  
  for (i in 1:N)
    eta[i] = X[i]*(beta_mu + beta_scen[Scen[i]] + beta_subj[Subj[i]]);
}

model {
    Y ~ bernoulli_logit(eta);
    
    beta_mu ~ normal(0, 2.5);
    sigma_scen ~ normal(0, 2.5);
    sigma_subj_resid ~ normal(0, 1);
    
    lambda_lower ~ normal(0, sigma_lambda);
    lambda_diag ~ normal(0, sigma_lambda);
    sigma_lambda ~ normal(0, 1);
    
    for (i in 1:Nsubj)
      beta_subj_raw[i] ~ normal(0., 1.);
    for (i in 1:Nscen)
      beta_scen_raw[i] ~ normal(0., 1.);
}

generated quantities {
  real log_lik[N];
  matrix[P,P] Rho;
  
  Rho = tcrossprod(Lambda);
  for (i in 1:N) log_lik[i] = bernoulli_logit_lpmf(Y[i] | eta[i]);
}

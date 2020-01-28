functions {
  real transferfunc(real x, vector w, vector loc, vector scale) {
    real y = 0;
    int K = num_elements(w);
    for (i in 1:K)
      y += w[i]*normal_cdf(x, loc[i], scale[i]);
    return y;
  }
}

data {
  int L;  // lower censoring
  int U;  // upper censoring
  int<lower=0> Nsub;  // number of subjects
  int<lower=0> Nc;  // number of cases
  int<lower=0> N;  // number of observations
  int<lower=0> P;  // number of regressors
  real<lower=L, upper=U> R[N];  // ratings
  int<lower=-1, upper=1> cens[N];  // -1 = left censor, 1 = right censor, 0 = none
  matrix[N, P] X;  // design matrix for data
  int<lower=0> S[N];  // subject corresponding to each rating
  int<lower=0> C[N];  // case corresponding to each rating
  int<lower=1> K;    //components of mixture transfer function
}

transformed data {
  real M;

  M = (U + L)/2.;
}
parameters {
  // mean for each regressor
  vector[P] mu;
  
  // variance across scenarios
  vector<lower=0>[P] eta;

  // variance across subjects
  vector<lower=0>[P] tau;

  // random effects
  vector[P] delta[Nc];  // scenario-specific
  vector[P] eps[Nsub];  // subject-specific
  
  real<lower=0> sigma;  // observation noise
  
  simplex[K] w_trans;   //component weights
  vector<lower=0>[K-1] l_trans_dist;     //component location distances
  vector<lower=0>[K] s_trans_raw; //component scales
  
}

transformed parameters {
  real theta[N];
  real z[N];
  vector[P] gamma[Nc];  // scenario effects
  vector[K] l_trans;
  vector[K] s_trans;
  real log_lik[N];
  vector[P] beta[Nsub, Nc];  // individual effects

  // draw scenario effects for each group
  for (c in 1:Nc)
    gamma[c] = mu + eta .* delta[c];

  // draw individual effects
  for (c in 1:Nc) 
    for (i in 1:Nsub) 
      beta[i, c] = gamma[c] + tau .* eps[i];
      
    
  if (K>1) {
    real sigmahat_trans; //mixture scale
      
    l_trans = append_row(0., cumulative_sum(l_trans_dist));
    l_trans -= dot_product(w_trans, l_trans);
    sigmahat_trans = dot_product(w_trans, l_trans .* l_trans + s_trans_raw .* s_trans_raw);
    l_trans ./= sqrt(sigmahat_trans);
    s_trans = s_trans_raw ./ sigmahat_trans;
  } else {
    l_trans = rep_vector(0.,1);
    s_trans = rep_vector(1.,1);
  }    
    
  // get linear predictor
  for (i in 1:N) {
    theta[i] = dot_product(X[i], beta[S[i],C[i]]);
    z[i] = transferfunc(theta[i], w_trans, l_trans, s_trans)*(U-L) + L;
      
    if (cens[i] == 0)
      log_lik[i] = normal_lpdf(R[i] | z[i], sigma);
    else if (cens[i] == -1)
      log_lik[i] = normal_lcdf(L | z[i], sigma);
    else if (cens[i] == 1)
      log_lik[i] = normal_lccdf(U | z[i], sigma);
    
  }
}

model {
  
  mu ~ normal(0, 2.5);
  eta ~ normal(0, 2.5);
  tau ~ normal(0, 2.5);
  sigma ~ normal(0, 10.);
  
  for (i in 1:Nsub)
    eps[i] ~ normal(0., 1.);
  for (c in 1:Nc) 
    delta[c] ~ normal(0., 1.);
    
  
  for (i in 1:N)
    target += log_lik[i];

  
  w_trans ~ dirichlet(rep_vector(1,K));
  l_trans_dist ~ normal(0, 2.5);
  s_trans_raw ~ normal(0, 2.5);
}

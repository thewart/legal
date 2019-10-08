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
  // vector<lower=0>[P] tau;

  // random effects
  vector[P] delta[Nc];  // scenario-specific
  // vector[P] eps[Nsub];  // subject-specific
  
  real<lower=0> sigma_r;  // observation noise
  
  simplex[K] w_trans;   //component weights
  vector<lower=0>[K-1] l_trans_dist;     //component location distances
  vector<lower=0>[K] s_trans; //component scales
  vector[N] u;
}

transformed parameters {
  real theta[N];
  vector[P] gamma[Nc];  // scenario effects
  vector[K] l_trans;
  // vector[P] beta[Nsub, Nc];  // individual effects

  // draw scenario effects for each group
  for (c in 1:Nc) {
    gamma[c] = mu + eta .* delta[c];
  }
  // draw individual effects
  // for (c in 1:Nc) {
  //   for (i in 1:Nsub) {
  //     beta[i, c] = gamma[c] + tau .* eps[i];
  //   }
  // }
  // get linear predictor
  for (i in 1:N) 
    theta[i] = dot_product(X[i], gamma[C[i]]);
  
  if (K>1) {
    l_trans = append_row(0, cumulative_sum(l_trans_dist));
    l_trans -= dot_product(w_trans, l_trans);
  } else {
    l_trans = rep_vector(0,1);
  }
}

model {
  
  for (i in 1:N)
    R[i] ~ normal(transferfunc(theta + u, w_trans, l_trans, s_trans), sigma_r) T[L, U];
  
  mu ~ normal(0, 1);
  eta ~ normal(0, 1);
  // tau ~ normal(0, M/4);
  
  // for (i in 1:Nsub)
  //   eps[i] ~ normal(0., 1.);
  for (c in 1:Nc)
    delta[c] ~ normal(0., 1.);
    
  sigma_r ~ normal(0, 10.);

  w_trans ~ dirichlet(rep_vector(1,K));
  l_trans_dist ~ normal(0, 1);
  s_trans ~ normal(0, 1);
  u ~ normal(0,1);
}

generated quantities {
  real log_lik[N];
  int ns;
  
  for (i in 1:N) {
    log_lik[i] = 0;
    for (j in 1:ns) {
      real z;
      z = transferfunc(theta + normal_rng(0,1), w_trans, l_trans, s_trans)
      log_lik[i] += normal_lcdf(R[i] | z, sigma_r);
    }
    log_lik[i] = log_lik[i]/ns - 
      log_diff_exp(normal_lcdf(L | 0, sigma_r), normal_lcdf(U | 0, sigma_r))
  }
  
}

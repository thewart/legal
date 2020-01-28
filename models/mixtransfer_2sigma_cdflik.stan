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
  int D; // distance between consecutive points on scale
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
  int<lower=1> ns; //number of samples for estimating pointwise predictive liklihood
}

transformed data {
  real M;
  real I;

  M = (U + L)/2.;
  I = D/2.;
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
  
  real<lower=1> sigma;  // observation noise
  
  simplex[K] w_trans;   //component weights
  vector<lower=0>[K-1] l_trans_dist;     //component location distances
  vector<lower=0>[K] s_trans; //component scales
  vector[N] u;
}

transformed parameters {
  vector[N] theta;
  vector[P] gamma[Nc];  // scenario effects
  vector[K] l_trans;
  vector[P] beta[Nsub, Nc];  // individual effects

  // draw scenario effects for each group
  for (c in 1:Nc) {
    gamma[c] = mu + eta .* delta[c];
  }
  // draw individual effects
  for (c in 1:Nc)
    for (i in 1:Nsub) 
      beta[i, c] = gamma[c] + tau .* eps[i];
      
  // get linear predictor
  for (i in 1:N) 
    theta[i] = dot_product(X[i], beta[S[i],C[i]]);
  
  if (K>1) {
    l_trans = append_row(0, cumulative_sum(l_trans_dist));
    l_trans -= dot_product(w_trans, l_trans);
  } else {
    l_trans = rep_vector(0,1);
  }
}

model {
  
  for (i in 1:N) {
    real z = transferfunc(theta[i] + u[i], w_trans, l_trans, s_trans)*(U-L) + L;
    if (cens[i] == 0)
      target += log_diff_exp(normal_lcdf(R[i]+I | z, sigma), normal_lcdf(R[i]-I | z, sigma));
    else if (cens[i] == -1)
      target += normal_lcdf(R[i]+I | z, sigma);
    else if (cens[i] == 1)
      target += normal_lccdf(R[i]-I | z, sigma);
    }

 mu ~ normal(0, 2.5);
 eta ~ normal(0, 2.5);
 tau ~ normal(0, 2.5);
 
 for (i in 1:Nsub)
   eps[i] ~ normal(0., 1.);
 for (c in 1:Nc)
   delta[c] ~ normal(0., 1.);
    
 sigma ~ normal(0, 10.);

 w_trans ~ dirichlet(rep_vector(1,K));
 l_trans_dist ~ normal(0, 2.5);
 s_trans ~ normal(0, 2.5);
 u ~ normal(0,1);
}

generated quantities {
  real log_lik[N];

  for (i in 1:N) {
    real lp[ns];

    for (j in 1:ns) {
      real z = transferfunc(theta[i] + normal_rng(0,1), w_trans, l_trans, s_trans)*(U-L) + L;
      if (cens[i] == 0)
        lp[j] = log_diff_exp(normal_lcdf(R[i]+I | z, sigma), normal_lcdf(R[i]-I | z, sigma));
      else if (cens[i] == -1)
        lp[j] = normal_lcdf(R[i]+I | z, sigma);
      else if (cens[i] == 1)
        lp[j] = normal_lccdf(R[i]-I | z, sigma);
      lp[j] -= log(ns);
    }

    log_lik[i] = log_sum_exp(lp);
  }
}

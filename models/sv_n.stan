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
}

transformed parameters {
  real theta[N];
  vector[P] gamma[Nc];  // scenario effects
  vector[P] beta[Nsub, Nc];  // individual effects

  // draw scenario effects for each group
  for (c in 1:Nc) {
    gamma[c] = mu + eta .* delta[c];
  }

  // draw individual effects
  for (c in 1:Nc) {
    for (i in 1:Nsub) {
      beta[i, c] = gamma[c] + tau .* eps[i];
    }
  }

  // get linear predictor
  for (j in 1:N)
    theta[j] = dot_product(X[j], beta[S[j], C[j]]);
}

model {
  
  mu ~ normal(M, M);
  eta ~ normal(0, M/4);
  tau ~ normal(0, M/4);
  
  for (i in 1:Nsub)
    eps[i] ~ normal(0., 1.);

  for (c in 1:Nc) {
    delta[c] ~ normal(0., 1.);
  }

  sigma ~ normal(0, M/4.);
  
  for (i in 1:N) {
      if (cens[i] == 0)
        R[i] ~ normal(theta[i], sigma);
      else if (cens[i] == -1)
        target += normal_lcdf(L | theta[i], sigma);
      else if (cens[i] == 1)
        target += normal_lccdf(U | theta[i], sigma);
  }
}

generated quantities {
  real Rhat[N];
  
  for (i in 1:N) {
    real pu;
    real pl;
    real Z;
    
    pl = normal_cdf(L, theta[i], sigma);
    pu = 1-normal_cdf(U, theta[i], sigma);
    Z = (exp(normal_lpdf(L | theta[i], sigma)) - exp(normal_lpdf(U | theta[i], sigma)))/(1-pu-pl);
    Rhat[i] = L*pl + U*pu + (theta[i] + Z*sigma^2)*(1-pl-pu);
  }
}

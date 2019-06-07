dat <- fread("~/code/legal/data/mri_subjects.csv")
dat[,witness:=factor(witness,levels=c("No Witness", "Yes Witness"))]
dat[,physical:=factor(physical, levels=c("No Physical", "Non-DNA", "DNA"))]
dat[,history:=factor(history, levels=c("No History", "Unrelated", "Related"))]
dat[,rating_type:=factor(rating_type,levels=unique(rating_type))]
bivar <- T

if (!bivar) dat <- dat[rating_type=="rating"]

Nsub <- length(unique(dat$uid))
Nc <- length(unique(dat$scenario))

# get upper and lower-bounded censored data
L <- min(dat$rating)
U <- max(dat$rating)

# get censoring data frame
R <- dat$rating
cens <- (R == U) - (R == L)

X <- model.matrix(~ 1 + physical + history + witness, data=dat)
S <- dat[,factor(uid) %>% as.numeric]
C <- dat$scenario

# useful dimensions
N <- dim(X)[1]
P <- dim(X)[2]
standat <- list(L=L, U=U, Nsub=Nsub, Nc=Nc, N=N, P=P, R=R, C=C,
                 X=X, S=S, cens=cens)
if (bivar) {
    standat <- c(standat,list(
      Ri=as.integer(as.factor(dat$rating_type)),
      Nr=length(unique(dat$rating_type))))
    initf <- function() 
      return(list(sigma=runif(2,10,50)))
    model <- stan_model("~/code/legal/models/mv_t.stan")
    
    out <- sampling(model,standat,chains=4,init=initf,
                    pars=c("mu","eta","tau","sigma","Omega","gamma"),iter=500,warmup=200)
    
} else {
  model <- stan_model("~/code/legal/models/sv_t.stan")
  initf <- function() 
    return(list(sigma=runif(1,10,50)))
  out <- sampling(model,standat,chains=3,init=initf,
                  pars=c("mu","eta","tau","sigma"),iter=500,warmup=200)
}
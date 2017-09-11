import scipy as sp
from scipy import linalg
from scipy.linalg.blas import dsyrk,dsymm,dgemm
eps = sp.finfo(sp.float64).eps
import time
from sklearn.datasets import load_digits

# Generate data
n,d=10000,200
X = sp.random.rand(n,d)*sp.random.rand(n,d)
print X.shape

##############################################################
# Covariance estimation
ts = time.time()
cov = sp.cov(X,rowvar=0)
duration = time.time()-ts
print ("Covariance proc. time: {}".format(duration))

m = sp.mean(X,axis=0,keepdims=True)
ts = time.time()
Xc = X-m
covs = dgemm(1.0/(n-1),Xc,Xc,trans_a=True,trans_b=False)
duration = time.time()-ts

print ("Fast Covariance proc. time: {}".format(duration))
print sp.mean((cov-covs)**2)

###############################################################

# Eigendecomposition
vp,Q = linalg.eigh(covs)
vp[vp<eps] = eps
Q /= sp.sqrt(vp)

# Compute full inverse
ts = time.time()
invCov = sp.triu(sp.dot(Q,Q.T))
duration = time.time()-ts
print ("Full inverse proc. time: {}".format(duration))

# Compute symmetric inverse
ts = time.time()
invCovs = dsyrk(1.,Q,trans=False)
durations = time.time()-ts
print ("Symmetric inverse proc. time: {}".format(durations))
print ("Speed-up: {}".format(duration/durations))
print sp.mean((invCov - invCovs)**2)
# print invCov
# print invCovs

###############################################################
invCov = sp.dot(Q,Q.T)
ts = time.time()
temp = sp.dot(Xc,invCov)
duration = time.time()-ts
print ("Full matrix product proc. time: {}".format(duration))

ts = time.time()
temps = dsymm(1,invCovs.T,Xc,side=1,lower=1)
durations = time.time()-ts
print ("Summetric matrix product proc. time: {}".format(durations))

ts = time.time()
tempss = dgemm(1,Xc,invCov)
durations = time.time()-ts
print ("DGEMM matrix product proc. time: {}".format(durations))

# scores = sp.sum(Xc*temp,axis=1)
# scoress = sp.sum(Xc*temps,axis=1)

print sp.mean((temp-temps)**2)
print sp.mean((temps-tempss)**2)

#####################################
newFeat = 10
featIdx = [1,4,3]
temp = dgemm(1.0,Qs[:,:],cov[newFeat,:][featIdx].T)
alpha= cov[newFeat,newFeat] - sp.sum(temp**2)
row_feat = dgemm(-1/alpha,Qs[:,:],temp)

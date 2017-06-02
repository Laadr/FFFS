# -*- coding: utf-8 -*-
import npfs as npfs
import scipy as sp
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# import HSIC_selection as hsic


n           = 40000 # Number of samples
d           = 100 # Number of dimension
d_info      = 4   # Number of informatives features
d_redundant = 0   # Number of redundant features

x, y = make_classification(n_samples=n, n_features=d, n_redundant=d_redundant, n_informative=d_info,n_classes=4,n_clusters_per_class=1,random_state=10,shuffle=False)
y += 1
# plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
# plt.show()

# Randomly permute features
generator = sp.random.seed(10)
indices   = sp.arange(d)
sp.random.shuffle(indices)
x[:, :]   = x[:, indices]
var       = sp.sort( [ sp.where(indices==k)[0][0] for k in xrange(d_info + d_redundant)] )

# Separate in train/test (80%/20%)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1, stratify=y)
del x,y

model    = npfs.GMMFeaturesSelection()
model.learn_gmm(xtrain, ytrain)
ts     = time.time()
yp       = model.predict_gmm(xtest)[0]
print "Processing time: ", time.time()-ts
yp.shape = ytest.shape
t        = sp.where(yp==ytest)[0]
print "Accuracy without selection: ", float(t.size)/ytest.size


# 5-CV
ts     = time.time()
idx,selectionOA,_ = model.selection('forward',xtrain, ytrain,criterion='kappa', varNb=4,nfold=5)
# idx.sort()
yp     = model.predict_gmm(xtest,featIdx=idx)[0]
j      = sp.where(yp.ravel()==ytest.ravel())[0]
OA     = (j.size*100.0)/ytest.size
print "\nResults\n"
print "Processing time: ", time.time()-ts
print "Selected features: ", idx
print "Evolution of accuracy during selection: ", selectionOA
print "\n Final accuracy: ", OA
print "Pertinent features (by construction): ", var

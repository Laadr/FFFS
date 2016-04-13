# -*- coding: utf-8 -*-
import npfs_v2 as npfs
import scipy as sp
import time

n     = 400 # Number of samples
d     = 20 # Number of dimension
noise = 0.2
sp.random.seed(10)
var   = sp.random.random_integers(0,d-1,3)

x               = sp.dot(sp.random.randn(n,d),sp.random.randn(d,d)) # Generate random samples
y               = sp.ones((n,1))
t               = 7*x[:,var[0]] -5.5*x[:,var[2]] + x[:,var[1]]**2
t += noise*sp.mean(t)*sp.random.randn(n) # add some noise
y[t>sp.mean(t)] = 2



model    = npfs.GMMFeaturesSelection()
model.learn_gmm(x, y)
yp       = model.predict_gmm(x,tau=None)[0]
yp.shape = y.shape
t        = sp.where(yp==y)[0]
print "Accuracy without selection: ", float(t.size)/y.size

# 5-CV
ts     = time.time()
idx,selectionOA = model.selection_cv('forward',x, y,criterion='accuracy', stopMethod='maxVar', delta=1.5, maxvar=0.2,nfold=5,balanced=True,tau=None,decisionMethod='inv')
idx.sort()
yp     = model.predict_gmm(x,featIdx=idx,tau=None)[0]
j      = sp.where(yp.ravel()==y.ravel())[0]
OA     = (j.size*100.0)/y.size
print "\nResults for 5-CV with accuracy as criterion and forward selection\n"
print "Processing time: ", time.time()-ts
print "Selected features: ", idx
print "Evolution of accuracy during selection: ", selectionOA
print "Final accuracy: ", OA
print "Pertinent features (by construction): ", var

# 5-CV
ts     = time.time()
idx,selectionOA = model.selection_cv('backward',x, y,criterion='accuracy', stopMethod='maxVar',delta=1.5, maxvar=0.2,nfold=5,balanced=True,tau=None,decisionMethod='inv')
# idx,selectionOA = model.backward_selection(x, y,criterion='accuracy',delta=1.5, maxvar=0.1,nfold=5,balanced=False,tau=None,decisionMethod='invUpdate')
idx.sort()
yp     = model.predict_gmm(x,featIdx=idx,tau=None)[0]
j      = sp.where(yp.ravel()==y.ravel())[0]
OA     = (j.size*100.0)/y.size
print "\nResults for 5-CV with accuracy as criterion and backward selection\n"
print "Processing time: ", time.time()-ts
print "Selected features: ", idx
print "Evolution of accuracy during selection: ", selectionOA
print "Final accuracy: ", OA
print "Pertinent features (by construction): ", var

# # 5-CV
# ts     = time.time()
# idx,selectionJM = model.forward_selection(x, y,criterion='JM',delta=1.5, maxvar=0.02,nfold=5,balanced=False,tau=0.0001)
# idx.sort()
# yp     = model.predict_gmm(x,featIdx=idx,tau=0.0001)[0]
# j      = sp.where(yp.ravel()==y.ravel())[0]
# OA     = (j.size*100.0)/y.size
# print "\nResults for 5-CV with JM distance as criterion\n"
# print "Processing time: ", time.time()-ts
# print "Selected features: ", idx
# print "Evolution of JM distance during selection: ", selectionJM
# print "Final accuracy: ", OA
# print "Pertinent features (by construction): ", var

# # LOO
# ts     = time.time()
# idx,selectionOA = model.forward_selection_loo(x, y,delta=1.5, maxvar=0.02,tau=0.0001)
# idx.sort()
# yp     = model.predict_gmm(x,featIdx=idx,tau=0.0001)[0]
# j      = sp.where(yp.ravel()==y.ravel())[0]
# OA     = (j.size*100.0)/y.size
# print "\nResults for LOO with accuracy as criterion\n"
# print "Processing time: ", time.time()-ts
# print "Selected features: ", idx
# print "Evolution of accuracy during selection: ", selectionOA
# print "Final accuracy: ", OA
# print "Pertinent features (by construction): ", var

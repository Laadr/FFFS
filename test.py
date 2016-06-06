# -*- coding: utf-8 -*-
import npfs as npfs
import scipy as sp
import time
import HSIC_selection as hsic

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
idx,selectionOA = model.selection('forward',x, y,criterion='accuracy', stopMethod='maxVar', delta=1.5, maxvar=3,nfold=2,balanced=True)
yp     = model.predict_gmm(x,featIdx=idx.sort(),tau=None)[0]
j      = sp.where(yp.ravel()==y.ravel())[0]
OA     = (j.size*100.0)/y.size
print "\nResults for 5-CV with accuracy as criterion and forward selection\n"
print "Processing time: ", time.time()-ts
print "Selected features: ", idx
print "Evolution of accuracy during selection: ", selectionOA
print "Final accuracy: ", OA
print "Pertinent features (by construction): ", var

# 5-CV
# ts     = time.time()
# idx    = hsic.HSIC_selection(x,y,maxvar=None)
# print idx
# yp     = model.predict_gmm(x,featIdx=idx[-3:].sort(),tau=None)[0]
# j      = sp.where(yp.ravel()==y.ravel())[0]
# OA     = (j.size*100.0)/y.size
# print "\nResults for 5-CV with accuracy as criterion and forward selection\n"
# print "Processing time: ", time.time()-ts
# print "Selected features: ", idx[-3:]
# print "Final accuracy: ", OA
# print "Pertinent features (by construction): ", var

# -*- coding: utf-8 -*-
import npfs as npfs
import scipy as sp
from sklearn.cross_validation import train_test_split
import time
import pickle

sp.set_printoptions(precision=3)

data_name      = 'aisa'
data = sp.load('Data/'+data_name+'.npz')
X    = data['X']
y    = data['Y']
C    = len(sp.unique(y))
print "Nb of samples: ",X.shape[0]," Nb of features: ",X.shape[1],"Nb of classes: ",C,"\n"


# Nt             = 50 # Nb of samples per class in training set
ntrial         = 30
# criterion      = 'accuracy'
# method         = 'SFFS' #'SFFS' or 'forward'
# stratification = False

Nts        = [(50,False), (100,False), (200,False), (400,False), (0.1,True), (0.3,True)] # Nb of samples per class in training set
methods    = ['SFFS', 'forward']
criterions = ['accuracy', 'kappa', 'F1Mean', 'JM', 'divKL']

for Nt,stratification in Nts:
    for criterion in criterions:
        for method in methods:

            if stratification:
                xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=Nt, random_state=1, stratify=y)
            else:
                sp.random.seed(0)
                xtrain = sp.empty((0,X.shape[1]))
                xtest  = sp.empty((0,X.shape[1]))
                ytrain = sp.empty((0,1))
                ytest  = sp.empty((0,1))
                for i in xrange(C):
                    t  = sp.where((i+1)==y)[0]
                    nc = t.size
                    rp = sp.random.permutation(nc)
                    xtrain = sp.concatenate( (X[t[rp[:Nt]],:], xtrain) )
                    xtest  = sp.concatenate( (X[t[rp[Nt:]],:], xtest) )
                    ytrain = sp.concatenate( (y[t[rp[:Nt]]], ytrain) )
                    ytest  = sp.concatenate( (y[t[rp[Nt:]]], ytest) )

            print "Nb of training samples: ",ytrain.shape[0]," Nb of test samples: ",ytest.shape[0],"\n"

            model    = npfs.GMMFeaturesSelection()
            model.learn_gmm(xtrain, ytrain)
            print "Proportion of classes (%): ", 100*model.prop.ravel(), "\n"
            yp       = model.predict_gmm(xtest,tau=None,decisionMethod='inv')[0]
            t        = sp.where(yp.ravel()==ytest.ravel())[0]
            print "Accuracy without selection: ", float(t.size)/ytest.size


            selected_idx = []
            confMatrix = npfs.ConfusionMatrix()
            for k in xrange(5,10):
                # 5-CV
                processingTime  = sp.zeros((ntrial,1))
                OA,kappa,F1Mean = sp.zeros((ntrial,1)), sp.zeros((ntrial,1)), sp.zeros((ntrial,1))
                idxs            = sp.zeros((ntrial,k))
                for i in xrange(ntrial):
                    # Change training set
                    if stratification:
                        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=Nt, random_state=1, stratify=y)
                    else:
                        sp.random.seed(0)
                        xtrain = sp.empty((0,X.shape[1]))
                        xtest  = sp.empty((0,X.shape[1]))
                        ytrain = sp.empty((0,1))
                        ytest  = sp.empty((0,1))
                        for i in xrange(C):
                            t  = sp.where((i+1)==y)[0]
                            nc = t.size
                            rp = sp.random.permutation(nc)
                            xtrain = sp.concatenate( (X[t[rp[:Nt]],:], xtrain) )
                            xtest  = sp.concatenate( (X[t[rp[Nt:]],:], xtest) )
                            ytrain = sp.concatenate( (y[t[rp[:Nt]]], ytrain) )
                            ytest  = sp.concatenate( (y[t[rp[Nt:]]], ytest) )
                    model.learn_gmm(xtrain, ytrain)

                    ts = time.time()
                    idx,selectionOA = model.selection(method,xtrain,ytrain,criterion=criterion,stopMethod='maxVar',delta=1.5,maxvar=k,nfold=5,balanced=True,tau=None,decisionMethod='inv',random_state=1)
                    processingTime[i,0] = time.time()-ts

                    idxs[i,:] = sp.asarray(idx)
                    yp        = model.predict_gmm(xtest,featIdx=idx,tau=None)[0]

                    confMatrix.compute_confusion_matrix(yp,ytest)
                    OA[i,0]     = confMatrix.get_OA()*100
                    kappa[i,0]  = confMatrix.get_kappa()
                    F1Mean[i,0] = confMatrix.get_F1Mean()

                selected_idx.append((idxs,OA,kappa,F1Mean,processingTime))
                print "\nResults for 5-CV with accuracy as criterion and " + method + " selection with " + str(k) + " variables\n"
                print "Processing time: ", sp.mean(processingTime)
                print "Selected features and accuracy: \n", sp.append(idxs,OA,axis=1)

                meanOA    = sp.mean(OA)
                meanKappa = sp.mean(kappa)
                meanF1    = sp.mean(F1Mean)
                print "Final accuracy: ", meanOA, " (std deviation: ", sp.sqrt( sp.mean(sp.square( (OA-meanOA) )) ), ")"
                print "Final kappa: ", meanKappa, " (std deviation: ", sp.sqrt( sp.mean(sp.square( (kappa-meanKappa) )) ), ")"
                print "Final F1-score mean: ", meanF1, " (std deviation: ", sp.sqrt( sp.mean(sp.square( (F1Mean-meanF1) )) ), ")"


            if stratification:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)+'_'+criterion
            else:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)+'_'+criterion


            f = open(filename+'.pckl', 'w')
            pickle.dump(selected_idx, f)
            f.close()

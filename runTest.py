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
methods    = ['forward','SFFS']
criterions = ['accuracy', 'kappa', 'F1Mean', 'JM', 'divKL']

for Nt,stratification in Nts:
    for criterion in criterions:
        for method in methods:

            if stratification:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)+'_'+criterion
                xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=Nt, random_state=1, stratify=y)
            else:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)+'_'+criterion
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

            printFile = open('Output/'+filename+'.txt','w')

            printFile.write("Nb of training samples: "+str(ytrain.shape[0])+" Nb of test samples: "+str(ytest.shape[0])+"\n\n")

            model    = npfs.GMMFeaturesSelection()
            model.learn_gmm(xtrain, ytrain)
            printFile.write("Proportion of classes (%): "+str(100*model.prop.ravel())+"\n\n")
            yp       = model.predict_gmm(xtest,tau=None,decisionMethod='inv')[0]
            t        = sp.where(yp.ravel()==ytest.ravel())[0]
            printFile.write("Accuracy without selection: "+str(float(t.size)/ytest.size)+"\n")


            selected_idx = []
            confMatrix = npfs.ConfusionMatrix()
            for k in xrange(5,8):
                # 5-CV
                processingTime  = sp.zeros((ntrial,1))
                OA,kappa,F1Mean = sp.zeros((ntrial,1)), sp.zeros((ntrial,1)), sp.zeros((ntrial,1))
                idxs            = sp.zeros((ntrial,k))
                for i in xrange(ntrial):
                    # Change training set
                    if stratification:
                        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=Nt, random_state=1, stratify=y)
                    else:
                        sp.random.seed(i)
                        xtrain = sp.empty((0,X.shape[1]))
                        xtest  = sp.empty((0,X.shape[1]))
                        ytrain = sp.empty((0,1))
                        ytest  = sp.empty((0,1))
                        for j in xrange(C):
                            t  = sp.where((j+1)==y)[0]
                            nc = t.size
                            rp = sp.random.permutation(nc)
                            xtrain = sp.concatenate( (X[t[rp[:Nt]],:], xtrain) )
                            xtest  = sp.concatenate( (X[t[rp[Nt:]],:], xtest) )
                            ytrain = sp.concatenate( (y[t[rp[:Nt]]], ytrain) )
                            ytest  = sp.concatenate( (y[t[rp[Nt:]]], ytest) )
                    model.learn_gmm(xtrain, ytrain)

                    ts = time.time()
                    print method,xtrain,ytrain,criterion
                    idx,selectionOA = model.selection(method,xtrain,ytrain,criterion=criterion,stopMethod='maxVar',delta=1.5,maxvar=k,nfold=5,balanced=True,tau=None,decisionMethod='inv',random_state=1)
                    print idx
                    processingTime[i,0] = time.time()-ts

                    idxs[i,:] = sp.asarray(idx)
                    yp        = model.predict_gmm(xtest,featIdx=idx,tau=None)[0]

                    confMatrix.compute_confusion_matrix(yp,ytest)
                    OA[i,0]     = confMatrix.get_OA()*100
                    kappa[i,0]  = confMatrix.get_kappa()
                    F1Mean[i,0] = confMatrix.get_F1Mean()

                selected_idx.append((idxs,OA,kappa,F1Mean,processingTime))
                printFile.write("\nResults for 5-CV with accuracy as criterion and " + method + " selection with " + str(k) + " variables\n\n")
                printFile.write("Processing time: "+str(sp.mean(processingTime))+"\n")
                printFile.write("Selected features and accuracy: \n"+str(idxs)+"\n")

                meanOA    = sp.mean(OA)
                meanKappa = sp.mean(kappa)
                meanF1    = sp.mean(F1Mean)
                printFile.write("Final accuracy: "+str(meanOA)+" (std deviation: "+str(sp.sqrt( sp.mean(sp.square( (OA-meanOA) )) ) )+")\n")
                printFile.write("Final kappa: "+str(meanKappa)+" (std deviation: "+str(sp.sqrt( sp.mean(sp.square( (kappa-meanKappa) )) ) )+")\n")
                printFile.write("Final F1-score mean: "+str(meanF1)+" (std deviation: "+str(sp.sqrt( sp.mean(sp.square( (F1Mean-meanF1) )) ) )+")\n")

            printFile.close()

            f = open('Results/'+filename+'.pckl', 'w')
            pickle.dump(selected_idx, f)
            f.close()

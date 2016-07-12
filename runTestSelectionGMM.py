# -*- coding: utf-8 -*-
import npfs as npfs
import scipy as sp
from sklearn.cross_validation import train_test_split, StratifiedKFold
import time
import pickle

sp.set_printoptions(precision=10)

data_name      = 'aisa'
data = sp.load('Data/'+data_name+'.npz')
X    = data['X']
y    = data['Y']
C    = len(sp.unique(y))
print "Nb of samples: ",X.shape[0]," Nb of features: ",X.shape[1],"Nb of classes: ",C,"\n"


ntrial = 1
minVar = 2
maxVar = 10

Nts        = [(50,False)]#, (100,False), (200,False), (0.005,True), (0.01,True), (0.025,True)] # Nb of samples per class in training set
methods    = ['forward','SFFS']
criterions = ['kappa']#['accuracy', 'kappa', 'F1Mean','JM', 'divKL']

for Nt,stratification in Nts:
    for criterion in criterions:
        for method in methods:

            if stratification:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)+'_'+criterion
            else:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)+'_'+criterion

            printFile = open('Output/'+filename+'_gmm.txt','w')

            results = []
            confMatrix = npfs.ConfusionMatrix()
            model = npfs.GMMFeaturesSelection()

            for i in xrange(ntrial):
                processingTime  = 0.
                OA,kappa,F1Mean = sp.zeros((maxVar-minVar+1,1)), sp.zeros((maxVar-minVar+1,1)), sp.zeros((maxVar-minVar+1,1))

                # Change training set
                if stratification:
                    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=Nt, random_state=i, stratify=y)
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
                idx,criterionEvolution,bestSets = model.selection(method,xtrain,ytrain,criterion=criterion,varNb=maxVar,nfold=5,random_state=0)
                processingTime = time.time()-ts

                for k in xrange(minVar,maxVar+1):
                    if method == 'SFFS':
                        yp = model.predict_gmm(xtest,featIdx=bestSets[k])[0]
                    else:
                        yp = model.predict_gmm(xtest,featIdx=idx[:k])[0]

                    confMatrix.compute_confusion_matrix(yp,ytest)
                    OA[k-minVar,0]     = confMatrix.get_OA()*100
                    kappa[k-minVar,0]  = confMatrix.get_kappa()
                    F1Mean[k-minVar,0] = confMatrix.get_F1Mean()

                if method == 'SFFS':
                    results.append((sp.asarray(idx),OA,kappa,F1Mean,processingTime,criterionEvolution,bestSets))
                else:
                    results.append((sp.asarray(idx),OA,kappa,F1Mean,processingTime,criterionEvolution))
                printFile.write("\nResults for 5-CV with " + criterion + " as criterion and " + method + " selection \n\n")
                printFile.write("Processing time: "+str(processingTime)+"\n")
                printFile.write("Selected features and accuracy: \n"+str(idx)+"\n")

                printFile.write("Final accuracy: "+str(OA)+"\n")
                printFile.write("Final kappa: "+str(kappa)+"\n")
                printFile.write("Final F1-score mean: "+str(F1Mean)+"\n")

            printFile.close()

            f = open('Results/'+filename+'_gmm.pckl', 'w')
            pickle.dump(results, f)
            f.close()
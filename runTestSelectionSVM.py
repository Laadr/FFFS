# -*- coding: utf-8 -*-
import npfs as npfs
import scipy as sp
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
import time
import pickle

sp.set_printoptions(precision=3)

data_name      = 'aisa'
data = sp.load('Data/'+data_name+'.npz')
X    = data['X']
y    = data['Y']
C    = len(sp.unique(y))
print "Nb of samples: ",X.shape[0]," Nb of features: ",X.shape[1],"Nb of classes: ",C,"\n"


ntrial = 2
minVar = 2
maxVar = 8

Nts        = [(50,False)]#, (100,False), (200,False), (400,False), (0.005,True), (0.01,True), (0.025,True)] # Nb of samples per class in training set
methods    = ['forward','SFFS']
criterions = ['accuracy', 'kappa', 'F1Mean','JM', 'divKL']

param_grid_svm = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]



for Nt,stratification in Nts:
    for criterion in criterions:
        for method in methods:

            if stratification:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)+'_'+criterion
            else:
                filename = data_name+'_'+method+'_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)+'_'+criterion

            f = open('Results/'+filename + '_gmm.pckl')
            res_gmm = pickle.load(f)
            f.close()

            printFile = open('Output/'+filename+'.txt','w')

            printFile.write("Nb of training samples: "+str(ytrain.shape[0])+" Nb of test samples: "+str(ytest.shape[0])+"\n\n")

            results = []
            confMatrix = npfs.ConfusionMatrix()

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

                idx = res_gmm[i][0]

                # SVM
                tt = time.time()
                scaler = preprocessing.StandardScaler().fit(xtrain)
                xtrain = scaler.transform(xtrain)
                cv = StratifiedKFold(ytrain.ravel(), n_folds=5, shuffle=True, random_state=0)
                grid = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=cv,n_jobs=-1,refit=False)
                grid.fit(xtrain[:,idx[:maxVar]], ytrain.ravel())
                clf = grid.best_estimator_
                processingTime = time.time()-tt

                xtest = scaler.transform(xtest)
                for k in xrange(minVar,maxVar+1):
                    clf.fit(xtrain[:,idx[:k]],ytrain.ravel())
                    yp = clf.predict(xtest[:,idx[:k]]).reshape(ytest.shape)

                    confMatrix.compute_confusion_matrix(yp,ytest)
                    OA[k-minVar,0]     = confMatrix.get_OA()*100
                    kappa[k-minVar,0]  = confMatrix.get_kappa()
                    F1Mean[k-minVar,0] = confMatrix.get_F1Mean()

                results.append((idx,OA,kappa,F1Mean,processingTime))
                printFile.write("\nResults for 5-CV with " + criterion + " as criterion and " + method + " selection \n\n")
                printFile.write("Processing time: "+str(processingTime)+"\n")
                printFile.write("Selected features and accuracy: \n"+str(idx)+"\n")

                printFile.write("Final accuracy: "+str(OA)+"\n")
                printFile.write("Final kappa: "+str(kappa)+"\n")
                printFile.write("Final F1-score mean: "+str(F1Mean)+"\n")

            printFile.close()

            f = open('Results/'+filename+'_svm.pckl', 'w')
            pickle.dump(results, f)
            f.close()
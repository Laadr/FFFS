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


ntrial = 20

Nts = [(50,False), (100,False), (200,False), (0.005,True), (0.01,True), (0.025,True)] # Nb of samples per class in training set

param_grid_gmm = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

svmTest    = True
param_grid_svm = [
  {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

for Nt,stratification in Nts:

    if stratification:
        filename = data_name+'_full_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)
    else:
        filename = data_name+'_full_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)

    printFile = open('Output/'+filename+'.txt','w')

    model      = npfs.GMM()
    confMatrix = npfs.ConfusionMatrix()
    results = []
    if svmTest:
        results_svm = []

    for i in xrange(ntrial):
        processingTime  = [0., 0.]
        OA,kappa,F1Mean = 0., 0., 0.
        if svmTest:
            processingTime_svm  = [0., 0.]
            OA_svm,kappa_svm,F1Mean_svm = 0.,0.,0.

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

        ## GMM
        # Training
        ts = time.time()
        cv = StratifiedKFold(ytrain.ravel(), n_folds=5, shuffle=True, random_state=0)
        kappaGrid = sp.zeros(len(param_grid_gmm))
        for k, (trainInd,testInd) in enumerate(cv):
            model.learn_gmm(xtrain[trainInd,:], ytrain[trainInd])

            for i, tau in enumerate(param_grid_gmm):
                yp = model.predict_gmm(xtrain[testInd,:], tau=tau)[0]

                confMatrix.compute_confusion_matrix(yp,ytrain[testInd])
                kappaGrid[i] += confMatrix.get_kappa()
        tau = param_grid_gmm[sp.argmax(kappaGrid)]
        param = tau
        model.learn_gmm(xtrain, ytrain)
        processingTime[0] = time.time()-ts

        # Prediction
        ts = time.time()
        yp = model.predict_gmm(xtest, tau=tau)[0]
        processingTime[1] = time.time()-ts

        # Evaluation of results
        confMatrix.compute_confusion_matrix(yp,ytest)
        OA     = confMatrix.get_OA()*100
        kappa  = confMatrix.get_kappa()
        F1Mean = confMatrix.get_F1Mean()


        ## SVM
        if svmTest:
            # Training
            tt = time.time()
            scaler = preprocessing.StandardScaler().fit(xtrain)
            xtrain = scaler.transform(xtrain)
            cv = StratifiedKFold(ytrain.ravel(), n_folds=5, shuffle=True, random_state=0)
            grid = GridSearchCV(SVC(), param_grid=param_grid_svm, cv=cv,n_jobs=-1)
            grid.fit(xtrain,ytrain.ravel())
            param_svm = grid.best_params_
            clf = grid.best_estimator_
            processingTime_svm[0] = time.time()-tt

            # Prediction
            tt = time.time()
            xtest = scaler.transform(xtest)
            yp = clf.predict(xtest).reshape(ytest.shape)
            processingTime_svm[1] = time.time()-tt

            # Evaluation of results
            confMatrix.compute_confusion_matrix(yp,ytest)
            OA_svm     = confMatrix.get_OA()*100
            kappa_svm  = confMatrix.get_kappa()
            F1Mean_svm = confMatrix.get_F1Mean()

        results.append((OA,kappa,F1Mean,processingTime,tau))
        printFile.write("\nResults with all features and GMM \n\n")
        printFile.write("Processing time: "+str(processingTime)+"\n")
        printFile.write("Tau: "+str(param)+"\n")
        printFile.write("Final accuracy: "+str(OA)+"\n")
        printFile.write("Final kappa: "+str(kappa)+"\n")
        printFile.write("Final F1-score mean: "+str(F1Mean)+"\n")

        if svmTest:
            results_svm.append((OA_svm,kappa_svm,F1Mean_svm,processingTime_svm,param_svm))
            printFile.write("\nResults with all features and SVM \n\n")
            printFile.write("Processing time: "+str(processingTime_svm)+"\n")
            printFile.write("Parameters: "+str(param_svm)+"\n")
            printFile.write("Final accuracy: "+str(OA_svm)+"\n")
            printFile.write("Final kappa: "+str(kappa_svm)+"\n")
            printFile.write("Final F1-score mean: "+str(F1Mean_svm)+"\n")

    printFile.close()

    f = open('Results/'+filename+'_gmm.pckl', 'w')
    pickle.dump(results, f)
    f.close()
    if svmTest:
        f = open('Results/'+filename+'_svm.pckl', 'w')
        pickle.dump(results_svm, f)
        f.close()
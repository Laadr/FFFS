# -*- coding: utf-8 -*-
'''
Created on 6 april 2016

@author: adlagrange
'''

import scipy as sp
from scipy import linalg
import multiprocessing as mp
import sklearn.cross_validation as cv
from sklearn.metrics import confusion_matrix

## Utilitary functions

def compute_metric_gmm(direction, criterion, variables, model_cv, samples, labels, idx):
    """
        Function that computes the accuracy of the model_cv using the variables : idx +/- one of variables
        Inputs:
            direction:      'backward' or 'forward' or 'SFFS'
            criterion:      criterion function use to discriminate variables
            variables:      the variable to add or delete to idx
            model_cv:       the model build with all the variables
            samples,labels: the samples/label for testing
            idx:            the pool of retained variables
        Output:
            metric: the estimated metric

        Used in GMM.forward_selection(), GMM.backward_selection()
    """
    variables  = sp.sort(variables)
    metric     = sp.zeros(variables.size)
    confMatrix = ConfusionMatrix()
    idx.sort()

    # Compute inv of covariance matrix
    if len(idx)==0:
        invCov = None
        logdet = None
    else:
        invCov     = sp.empty((model_cv.C,len(idx),len(idx)))
        logdet     = sp.empty((model_cv.C))

        for c in xrange(model_cv.C):
            vp,Q,rcond    = model_cv.decomposition(model_cv.cov[c,idx,:][:,idx])
            invCov[c,:,:] = sp.dot(Q,((1/vp)*Q).T)
            invCov[c,:,:] = linalg.inv(model_cv.cov[c,idx,:][:,idx])
            logdet[c]     = sp.sum(sp.log(vp))

    for i,var in enumerate(variables):
        predLabels = model_cv.predict_gmm_update(direction,samples,invCov,logdet,(i,var),featIdx=idx)[0]

        confMatrix.compute_confusion_matrix(predLabels,labels)
        if criterion=='accuracy':
            metric[i] = confMatrix.get_OA()
        elif criterion=='F1Mean':
            metric[i] = confMatrix.get_F1Mean()
        elif criterion=='kappa':
            metric[i] = confMatrix.get_kappa()

    return metric

def compute_JM(direction, variables, model, idx):
    """
        Function that computes the Jeffries–Matusita distance of the model_cv using the variables : idx +/- one of variables
        Inputs:
            variables: the variable to add to idx
            model_cv:  the model build with all the variables
            idx:       the pool of retained variables
        Output:
            JM: the estimated Jeffries–Matusita distance

        Used in GMM.forward_selection() and GMM.backward_selection()
    """
    # Initialization
    JM = sp.zeros(variables.size)
    d  = sp.zeros((model.C,variables.size))

    # Cast and sort index of selected variables
    idx.sort()

    # Compute all possible update of det cov(idx)
    if len(idx)==0:
        for c in xrange(model.C):
            for k,var in enumerate(variables):
                d[c,k] = model.cov[c,var,var]
    else:
        for c in xrange(model.C):
            vp,Q,rcond = model.decomposition(model.cov[c,idx,:][:,idx])
            det = sp.prod(vp)
            invCov = sp.dot(Q,((1/vp)*Q).T)
            for k,var in enumerate(variables):
                if direction=='forward':
                    maj_cst = model.cov[c,var,var] - sp.dot(model.cov[c,var,:][idx], sp.dot(invCov,model.cov[c,var,:][idx].T) )
                elif direction=='backward':
                    maj_cst = 1/float( invCov[k,k] )
                d[c,k]  = maj_cst * det
        del vp,Q,rcond,maj_cst,invCov

    if len(idx)==0:
        for i in xrange(model.C):
            for j in xrange(i+1,model.C):
                for k,var in enumerate(variables):
                    md     = (model.mean[i,var]-model.mean[j,var])
                    cs     = (model.cov[i,var,var]+model.cov[j,var,var])/2

                    dij    = cs
                    invCov = 1/float(cs)

                    bij    = sp.dot(md, sp.dot(invCov,md.T) )/8 + 0.5*sp.log(dij/sp.sqrt(d[i,k]*d[j,k]))
                    JM[k]  += sp.sqrt(2*(1-sp.exp(-bij)))*model.prop[i]*model.prop[j]

    else:
        for i in xrange(model.C):
            for j in xrange(i+1,model.C):
                cs  = (model.cov[i,idx,:][:,idx]+model.cov[j,idx,:][:,idx])/2
                vp,Q,rcond = model.decomposition(cs)
                invCov = sp.dot(Q,((1/vp)*Q).T)
                det = sp.prod(vp)

                for k,var in enumerate(variables):
                    md      = (model.mean[i,idx]-model.mean[j,idx])

                    if direction=='forward':
                        id_t = list(idx)
                        id_t.append(var)
                        id_t.sort()

                        c1 = (model.cov[i,var,var]+model.cov[j,var,var])/2
                        c2 = (model.cov[i,var,:][idx]+model.cov[j,var,:][idx])/2
                        maj_cst = c1 - sp.dot(c2, sp.dot(invCov,c2.T) )
                        dij = maj_cst * det

                        md_new  = (model.mean[i,id_t]-model.mean[j,id_t])
                        ind      = id_t.index(var)
                        row_feat = -1/float(maj_cst) * sp.dot(c2,invCov)
                        row_feat = sp.insert( row_feat, ind, 1/float(maj_cst) )
                        cst_feat = maj_cst * (sp.dot(row_feat,md_new.T)**2)

                    elif direction=='backward':
                        maj_cst     = 1/float( invCov[k,k] )
                        dij = maj_cst * det

                        row_feat   = invCov[k,:]
                        cst_feat   = - maj_cst * (sp.dot(row_feat,md.T)**2)

                    temp = sp.dot(md, sp.dot(invCov,md.T) ) + cst_feat

                    bij = temp/8 + 0.5*sp.log(dij/sp.sqrt(d[i,k]*d[j,k]))
                    JM[k] += sp.sqrt(2*(1-sp.exp(-bij)))*model.prop[i]*model.prop[j]

    return JM

def compute_divKL(direction, variables, model, idx):
    """
        Function that computes the  Kullback–Leibler divergence of the model_cv using the variables : idx +/- one of variables
        Inputs:
            variables: the variable to add to idx
            model_cv:  the model build with all the variables
            idx:       the pool of retained variables
        Output:
            divKL: the estimated Kullback–Leibler divergence

        Used in GMM.forward_selection() and GMM.backward_selection()
    """
    # Initialization
    divKL  = sp.zeros(variables.size)
    invCov = sp.empty((model.C,len(idx),len(idx)))

    # Cast and sort index of selected variables
    idx.sort()

    if len(idx)==0:
        for k,var in enumerate(variables):
            for i in xrange(model.C):
                invCovI = 1/float(model.cov[i,var,var])
                for j in xrange(i+1,model.C):
                    invCovJ = 1/float(model.cov[j,var,var])

                    md  = (model.mean[i,var]-model.mean[j,var])
                    divKL[k] += 0.25*( invCovI*model.cov[j,var,var] + invCovJ*model.cov[i,var,var] + md*(invCovI + invCovJ)*md ) * model.prop[i]*model.prop[j]
    else:
        # Compute invcov de idx
        for c in xrange(model.C):
            vp,Q,rcond = model.decomposition(model.cov[c,idx,:][:,idx])
            invCov[c,:,:] = sp.dot(Q,((1/vp)*Q).T)
        del vp,Q,rcond

        if direction=='forward':
            invCov_update = sp.empty((model.C,len(idx)+1,len(idx)+1))
        elif direction=='backward':
            invCov_update = sp.empty((model.C,len(idx)-1,len(idx)-1))

        for k,var in enumerate(variables):
            if direction=='forward':
                id_t      = list(idx)
                id_t.append(var)
                id_t.sort()
                mask      = sp.ones(len(id_t), dtype=bool)
                ind       = id_t.index(var)
                mask[ind] = False
            elif direction=='backward':
                id_t    = list(idx)
                mask    = sp.ones(len(idx), dtype=bool)
                mask[k] = False
                del id_t[k]

            for c in xrange(model.C):
                if direction=='forward':
                    maj_cst = model.cov[c,var,var] - sp.dot(model.cov[c,var,:][idx], sp.dot(invCov[c,:,:],model.cov[c,var,:][idx].T) )
                    invCov_update[c,mask,:][:,mask] = invCov[c,:,:] + 1/float(maj_cst) * sp.dot( sp.dot(invCov[c,:,:],model.cov[c,var,:][idx].T) , sp.dot(model.cov[c,var,:][idx],invCov[c,:,:]) )
                    invCov_update[c,ind,mask]  = - 1/float(maj_cst) * sp.dot(invCov[c,:,:],model.cov[c,var,:][idx].T)
                    invCov_update[c,mask,ind]  = - 1/float(maj_cst) * sp.dot(model.cov[c,var,:][idx],invCov[c,:,:])
                    invCov_update[c,ind,ind]   = 1/float(maj_cst)

                elif direction=='backward':
                    invCov_update[c,:,:] = invCov[c,mask,mask] - 1/float(invCov[c,k,k]) * sp.dot(model.cov[c,var,:][idx].T , model.cov[c,var,:][idx])


            for i in xrange(model.C):
                for j in xrange(i+1,model.C):
                    md  = (model.mean[i,id_t]-model.mean[j,id_t])
                    divKL[k] += 0.25*( sp.trace( sp.dot( invCov_update[j,:,:],model.cov[i,id_t,:][:,id_t] ) + sp.dot( invCov_update[i,:,:],model.cov[j,id_t,:][:,id_t] ) ) + sp.dot(md,sp.dot(invCov_update[j,:,:]+invCov_update[i,:,:],md.T)) ) * model.prop[i]*model.prop[j]

    return divKL

## Confusion matrix

class ConfusionMatrix(object):
    def __init__(self):
        self.confusion_matrix = None
        self.n = None

    def compute_confusion_matrix(self,yp,yr):
        """
            Compute the confusion matrix
            Inputs:
                yp: predicted labels
                yr: reference labels
        """
        # Initialization
        self.n                = yp.size
        # C                     = int(yr.max())
        self.confusionMatrix  = confusion_matrix(yr,yp)
        # self.confusionMatrix = sp.zeros((C,C),dtype=int)

        # # Compute confusion matrix
        # for c1 in xrange(C):
        #     tmp = (yp==(c1+1))
        #     for c2 in xrange(C):
        #         self.confusionMatrix[c1, c2] = sp.sum( tmp * (yr==(c2+1)) )

    def get_OA(self):
        """
            Compute overall accuracy
        """
        return sp.sum(sp.diag(self.confusionMatrix))/float(self.n)

    def get_kappa(self):
        """
            Compute Kappa
        """
        nl = sp.sum(self.confusionMatrix,axis=1)
        nc = sp.sum(self.confusionMatrix,axis=0)
        OA = sp.sum(sp.diag(self.confusionMatrix))/float(self.n)

        return ((self.n**2)*OA - sp.sum(nc*nl))/(self.n**2-sp.sum(nc*nl))

    def get_F1Mean(self):
        """
            Compute F1 Mean
        """
        nl = sp.sum(self.confusionMatrix,axis=1)
        nc = sp.sum(self.confusionMatrix,axis=0)
        return 2*sp.mean( sp.divide( sp.diag(self.confusionMatrix), (nl + nc)) )

## Gaussian Mixture Model

class GMM(object):

    def __init__(self, d=0, C=0):
        self.nbSpl = sp.empty((C,1))   # array of number of samples in each class
        self.prop  = sp.empty((C,1))   # array of proportion in training set
        self.mean  = sp.empty((C,d))   # array of means
        self.cov   = sp.empty((C,d,d)) # array of covariance matrices
        self.C     = C                 # number of class
        self.d     = d                 # number of features

        self.idxDecomp = []                # empty if no decomposition since last learning and store index of features of last decomposition in the case of selection
        self.vp        = sp.empty((C,d))   # array of eigenvalues
        self.Q         = sp.empty((C,d,d)) # array of eigenvectors

    def decomposition(self, M):
        """
            Compute the decompostion of symmetric matrix
            Inputs:
                M:   matrix to decompose
            Outputs:
                vp:    eigenvalues
                Q:     eigenvectors
                rcond: conditioning
        """
        # Decomposition
        vp,Q = linalg.eigh(M)

        # Compute conditioning
        eps = sp.finfo(sp.float64).eps
        if vp.max()<eps:
            rcond = 0
        else:
            rcond = vp.min()/vp.max()

        vp[vp<eps] = eps

        return vp,Q,rcond

    def learn_gmm(self, samples, labels):
        """
            Method that learns the GMM from training samples and store the mean, covariance and proportion of each class in class members.
            Input:
                samples: training samples
                labels:  training labels (must be exactly C labels between 1 and C)
        """
        # Get information from the data
        self.C = int(labels.max(0)) # Number of classes
        self.d = samples.shape[1]   # Number of variables

        # Initialization
        self.nbSpl     = sp.empty((self.C,1))   # Vector of number of samples for each class
        self.prop      = sp.empty((self.C,1))   # Vector of proportion
        self.mean      = sp.empty((self.C,self.d))   # Vector of means
        self.cov       = sp.empty((self.C,self.d,self.d)) # Matrix of covariance
        self.idxDecomp = [] # reset to empty to allow recomputation of eigenvalues
        self.vp        = sp.empty((self.C,samples.shape[1]))   # array of eigenvalues
        self.Q         = sp.empty((self.C,samples.shape[1],samples.shape[1])) # array of eigenvectors

        # Learn the parameter of the model for each class
        for i in xrange(self.C):
            # Get index of class i+1 samples
            j = sp.where(labels==(i+1))[0]

            # Update GMM
            self.nbSpl[i]   = float(j.size)
            self.mean[i,:]  = sp.mean(samples[j,:],axis=0)
            self.cov[i,:,:] = sp.cov(samples[j,:],rowvar=0) # implicit: with no bias

        self.prop = self.nbSpl/samples.shape[0]

    def predict_gmm(self, testSamples, tau=0):
        """
            Function that predict the label for testSamples using the learned model
            Inputs:
                testSamples: the samples to be classified
                tau:         regularization parameter
            Outputs:
                predLabels: the class
                scores:     the decision value for each class
        """
        # Get information from the data
        nbTestSpl = testSamples.shape[0] # Number of testing samples

        # Initialization
        scores = sp.empty((nbTestSpl,self.C))

        # Start the prediction for each class
        for c in xrange(self.C):
            testSamples_c = testSamples - self.mean[c,:]

            if self.idxDecomp == []:
                self.vp[c,:],self.Q[c,:,:],_ = self.decomposition(self.cov[c,:,:])

            regvp = self.vp[c,:] + tau

            logdet        = sp.sum(sp.log(regvp))
            cst           = logdet - 2*sp.log(self.prop[c]) # Pre compute the constant term

            temp = sp.dot( sp.dot(self.Q[c,:,:][:,:],((1/regvp)*self.Q[c,:,:][:,:]).T) , testSamples_c.T).T

            scores[:,c] = sp.sum(testSamples_c*temp,axis=1) + cst

            del temp,testSamples_c
        self.idxDecomp = range(testSamples.shape[1])

        # Assign the label to the minimum value of scores
        predLabels = sp.argmin(scores,1)+1

        return predLabels,scores


class GMMFeaturesSelection(GMM):

    def __init__(self, d=0, C=0):
        super(GMMFeaturesSelection, self).__init__(d,C)

    def predict_gmm(self, testSamples, featIdx=None, tau=0):
        """
            Function that predict the label for testSamples using the learned model
            Inputs:
                testSamples: the samples to be classified
                featIdx:     indices of features to use for classification
                tau:         regularization parameter
            Outputs:
                predLabels: the class
                scores:     the decision value for each class
        """
        # Get information from the data
        nbTestSpl = testSamples.shape[0] # Number of testing samples

        # Initialization
        scores = sp.empty((nbTestSpl,self.C))

        # If not specified, predict with all features
        if featIdx is None:
            idx = range(testSamples.shape[1])
        else:
            idx = featIdx

        # Allocate storage for decomposition in eigenvalues
        if self.idxDecomp != idx:
            self.vp    = sp.empty((self.C,len(idx)))   # array of eigenvalues
            self.Q     = sp.empty((self.C,len(idx),len(idx))) # array of eigenvectors

        # Start the prediction for each class
        for c in xrange(self.C):
            testSamples_c = testSamples[:,idx] - self.mean[c,idx]

            if self.idxDecomp != idx:
                self.vp[c,:],self.Q[c,:,:],_ = self.decomposition(self.cov[c,idx,:][:,idx])

            regvp = self.vp[c,:] + tau

            logdet        = sp.sum(sp.log(regvp))
            cst           = logdet - 2*sp.log(self.prop[c]) # Pre compute the constant term

            temp = sp.dot( sp.dot(self.Q[c,:,:][:,:],((1/regvp)*self.Q[c,:,:][:,:]).T) , testSamples_c.T).T

            scores[:,c] = sp.sum(testSamples_c*temp,axis=1) + cst

            del temp,testSamples_c
        self.idxDecomp = idx

        # Assign the label to the minimum value of scores
        predLabels = sp.argmin(scores,1)+1

        return predLabels,scores

    def predict_gmm_update(self, direction, testSamples, invCov, logdet, newFeat, featIdx=None):
        """
            Function that predict the label for testSamples using the learned model (with an update method of the inverse of covariance matrix)
            Inputs:
                direction:   'backward' or 'forward'
                testSamples: the samples to be classified
                invCov:      inverte of covariance matrix of already selected features
                newFeat:     couple (index,feature) to add or delete
                featIdx:     indices of features to use for classification
            Outputs:
                predLabels: the class
                scores:     the decision value for each class
        """
        # Get machine precision
        eps = sp.finfo(sp.float64).eps

        # Get information from the data
        nbTestSpl = testSamples.shape[0] # Number of testing samples

        # Initialization
        scores = sp.empty((nbTestSpl,self.C))

        # If not specified, predict with all features
        if featIdx is None:
            featIdx = range(testSamples.shape[1])

        # New set of features
        if direction=='forward':
            id_t = list(featIdx)
            id_t.append(newFeat[1])
        elif direction=='backward':
            id_t = list(featIdx)

        # Start the prediction for each class
        for c in xrange(self.C):
            testSamples_c = testSamples[:,id_t] - self.mean[c,id_t]

            if len(id_t)==1:
                scores[:,c] = sp.sum(testSamples_c*testSamples_c,axis=1)/self.cov[c,id_t,id_t] + sp.log(self.cov[c,id_t,id_t]) - self.logprop[c]

            else:
                if direction=='forward':
                    d_feat     = self.cov[c,newFeat[1],newFeat[1]] - sp.dot(self.cov[c,newFeat[1],:][featIdx], sp.dot(invCov[c,:,:][:,:],self.cov[c,newFeat[1],:][featIdx].T) )
                    if d_feat > eps:
                        logdet_update = sp.log(d_feat) + logdet[c]
                    else:
                        logdet_update = sp.log(eps) + logdet[c]

                    row_feat       = sp.empty((len(id_t)))
                    row_feat[:len(featIdx)] = -1/float(d_feat) * sp.dot(self.cov[c,:,newFeat[1]][featIdx],invCov[c,:,:][:,:])
                    row_feat[-1]  = 1/float(d_feat)
                    cst_feat       = d_feat * (sp.dot(row_feat,testSamples_c.T)**2)
                    testSamples_c  = testSamples[:,featIdx] - self.mean[c,featIdx]

                elif direction=='backward':
                    d_feat     = 1/float( invCov[c,newFeat[0],newFeat[0]] )
                    logdet_update = - sp.log(d_feat) + logdet[c]

                    row_feat   = invCov[c,newFeat[0],:]
                    cst_feat   = - d_feat * (sp.dot(row_feat,testSamples_c.T)**2)

                cst = logdet_update - self.logprop[c] # Pre compute the constant term

                temp = sp.dot(invCov[c,:,:][:,:], testSamples_c.T).T

                scores[:,c] = sp.sum(testSamples_c*temp,axis=1) + cst_feat + cst

                del temp

            del testSamples_c

        # Assign the label to the minimum value of scores
        predLabels = sp.argmin(scores,1)+1
        return predLabels,scores

    def selection(self, direction, samples, labels, criterion='accuracy', stopMethod='maxVar', delta=0.1, maxvar=0.2, nfold=5, balanced=True, ncpus=None, random_state=1):
        """
            Function that selects the most discriminative variables according to a given search method
            Inputs:
                direction:       'backward' or 'forward'
                samples, labels: the training samples and their labels
                criterion:       the criterion function to use for selection (accuracy, kappa, F1Mean, JM).  Default: 'accuracy'
                stopMethod:      the stopping criterion. It can be either 'maxVar' to continue until maxvar variables are selected or either 'variation' to continue until variation of criterion function are less than delta. Default: 'maxVar'
                delta :          the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta. Default value 0.1%
                maxvar:          maximum number of extracted variables. Default value: 20% of the original number
                nfold:           number of folds for the cross-validation. Default value: 5
                balanced:        If true, same proportion of each class in each fold. Default: True
                ncpus:           number of cpus to use for parallelization. Default: all

            Outputs:
                idx:              the selected variables
                criterionBestVal: the criterion value estimated for each idx by nfold-fold cv
        """
        # Get some information from the variables
        n = samples.shape[0]      # Number of samples

        # Cast to speed processing time
        labels = labels.ravel().astype(int)

        if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':

            # Creation of folds
            if balanced:
                kfold = cv.StratifiedKFold(labels.ravel(),n_folds=nfold,shuffle=True,random_state=random_state) # kfold is an iterator
            else:
                kfold = cv.KFold(n,n_folds=nfold,shuffle=True) # kfold is an iterator

            ## Pre-update the models
            model_pre_cv = [GMMFeaturesSelection(d=self.d, C=self.C) for i in xrange(len(kfold))]
            for k, (trainInd,testInd) in enumerate(kfold):
                # Get training data for this cv round
                testSamples,testLabels = samples[testInd,:], labels[testInd]
                nk = float(testLabels.size)

                # Update the model for each class
                for c in xrange(self.C):
                    classInd = sp.where(testLabels==(c+1))[0]
                    nk_c     = float(classInd.size)
                    mean_k   = sp.mean(testSamples[classInd,:],axis=0)
                    cov_k    = sp.cov(testSamples[classInd,:],rowvar=0)

                    model_pre_cv[k].nbSpl[c]  = self.nbSpl[c] - nk_c
                    model_pre_cv[k].mean[c,:] = (self.nbSpl[c]*self.mean[c,:]-nk_c*mean_k)/(self.nbSpl[c]-nk_c)
                    model_pre_cv[k].cov[c,:]  = ((self.nbSpl[c]-1)*self.cov[c,:,:] - (nk_c-1)*cov_k - nk_c*self.nbSpl[c]/model_pre_cv[k].nbSpl[c]*sp.outer(self.mean[c,:]-mean_k,self.mean[c,:]-mean_k))/(model_pre_cv[k].nbSpl[c]-1)

                    del classInd,nk_c,mean_k,cov_k

                # Update proportion
                model_pre_cv[k].prop = model_pre_cv[k].nbSpl/(n-nk)

                # Precompute cst
                model_pre_cv[k].logprop = 2*sp.log(model_pre_cv[k].prop)

                del testSamples,testLabels,nk
        else:
            kfold = None
            model_pre_cv = None

        if direction == 'forward':
            return self.forward_selection(samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, ncpus)
        elif direction == 'backward':
            return self.backward_selection(samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, ncpus)
        elif direction == 'SFFS':
            return self.floating_forward_selection(samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, ncpus)

    def forward_selection(self, samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, ncpus):
        """
            Function that selects the most discriminative variables according to a forward search
            Inputs:
                samples, labels: the training samples and their labels
                criterion:       the criterion function to use for selection (accuracy, kappa, F1Mean, JM, divKL).
                stopMethod:      the stopping criterion. It can be either 'maxVar' to continue until maxvar variables are selected or either 'variation' to continue until variation of criterion function are less than delta.
                delta:           the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta.
                maxvar:          maximum number of extracted variables.
                kfold:           k-folds for the cross-validation.
                model_pre_cv:    GMM models for each CV.
                ncpus:           number of cpus to use for parallelization.

            Outputs:
                idx:              the selected variables
                criterionBestVal: the criterion value estimated for each idx by nfold-fold cv
        """
        # Get some information from the variables
        n = samples.shape[0]      # Number of samples
        if ncpus is None:
            ncpus=mp.cpu_count() # Get the number of core

        # Initialization
        nbSelectFeat     = 0                       # Initialization of the counter
        variables        = sp.arange(self.d)       # At step zero: d variables available
        idx              = []                      # and no selected variable
        criterionBestVal = []                      # list of the evolution the OA estimation
        if maxvar==0.2:
            maxvar = sp.floor(self.d*maxvar) # Select at max maxvar % of the original number of variables

        # Start the forward search
        while(nbSelectFeat<maxvar) and (variables.size!=0):

            # Compute criterion function
            if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':
                # Parallelize cv
                pool = mp.Pool(processes=ncpus)
                processes =  [pool.apply_async(compute_metric_gmm, args=('forward',criterion,variables,model_pre_cv[k],samples[testInd,:],labels[testInd],idx)) for k, (trainInd,testInd) in enumerate(kfold)]
                pool.close()
                pool.join()

                # Compute mean criterion value over each processus
                criterionVal = sp.zeros(variables.size)
                for p in processes:
                    criterionVal += p.get()
                criterionVal /= len(kfold)
                del processes,pool

            elif criterion == 'JM':
                criterionVal =  compute_JM('forward',variables,self,idx)

            elif criterion == 'divKL':
                criterionVal =  compute_divKL('forward',variables,self,idx)


            # Select the variable that provides the highest loocv
            bestVar = sp.argmax(criterionVal)                # get the indice of the maximum of criterion values
            criterionBestVal.append(criterionVal[bestVar])         # save criterion value

            if nbSelectFeat==0:
                idx.append(variables[bestVar])           # add the selected variables to the pool
                variables = sp.delete(variables,bestVar)  # remove the selected variables from the initial set
            elif (stopMethod=='variation') and (((criterionBestVal[nbSelectFeat]-criterionBestVal[nbSelectFeat-1])/criterionBestVal[nbSelectFeat-1]*100) < delta):
                criterionBestVal.pop()
                break
            else:
                idx.append(variables[bestVar])
                variables=sp.delete(variables,bestVar)
            nbSelectFeat += 1

        ## Return the final value
        return idx,criterionBestVal

    def backward_selection(self, samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, ncpus):
        """
            Function that selects the most discriminative variables according to a backward search
            Inputs:
                samples, labels:  the training samples and their labels
                criterion:        the criterion function to use for selection (accuracy, kappa, F1Mean, JM, divKL).
                stopMethod:       the stopping criterion. It can be either 'maxVar' to continue until maxvar variables are selected or either 'variation' to continue until variation of criterion function are less than delta.
                delta :           the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta.
                maxvar:           maximum number of extracted variables.
                kfold:            k-folds for the cross-validation.
                model_pre_cv:     GMM models for each CV.
                ncpus:            number of cpus to use for parallelization.
            Outputs:
                idx:              the selected variables
                criterionBestVal: the criterion value estimated for each idx by nfold-fold cv
        """
        # Get some information from the variables
        n = samples.shape[0]      # Number of samples
        if ncpus is None:
            ncpus=mp.cpu_count() # Get the number of core

        # Initialization
        idx              = sp.arange(self.d)       # and no selected variable
        criterionBestVal = []                      # list of the evolution the OA estimation
        if maxvar==0.2:
            maxvar = sp.floor(self.d*maxvar) # Select at max maxvar % of the original number of variables

        # Start the forward search
        while(idx.size>maxvar):

            # Compute criterion function
            if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':
                # Parallelize cv
                pool = mp.Pool(processes=ncpus)
                processes =  [pool.apply_async(compute_metric_gmm, args=('backward',criterion,idx,model_pre_cv[k],samples[testInd,:],labels[testInd],idx)) for k, (trainInd,testInd) in enumerate(kfold)]
                pool.close()
                pool.join()

                # Compute mean criterion value over each processus
                criterionVal = sp.zeros(idx.size)
                for p in processes:
                    criterionVal += p.get()
                criterionVal /= len(kfold)
                del processes,pool

            elif criterion == 'JM':
                criterionVal = compute_JM('backward',idx,self,idx)

            elif criterion == 'divKL':
                criterionVal = compute_divKL('backward',idx,self,idx)


            # Select the variable that provides the highest loocv
            worstVar = sp.argmax(criterionVal)                # get the indice of the maximum of criterion values
            criterionBestVal.append(criterionVal[worstVar])    # save criterion value

            if (stopMethod=='variation') and (((criterionBestVal[-2]-criterionBestVal[-1])/criterionBestVal[-2]*100) < delta):
                criterionBestVal.pop()
                break
            else:
                mask           = sp.ones(len(idx), dtype=bool)
                mask[worstVar] = False
                idx            = idx[mask]                    # delete the selected variable of the pool

        ## Return the final value
        return idx,criterionBestVal

    def floating_forward_selection(self, samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, ncpus):
        """
            Function that selects the most discriminative variables according to a floating forward search
            Inputs:
                samples, labels:  the training samples and their labels
                criterion:        the criterion function to use for selection (accuracy, kappa, F1Mean, JM, divKL).
                stopMethod:       the stopping criterion. It can be either 'maxVar' to continue until maxvar variables are selected or either 'variation' to continue until variation of criterion function are less than delta.
                delta :           the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta.
                maxvar:           maximum number of extracted variables.
                kfold:            k-folds for the cross-validation.
                model_pre_cv:     GMM models for each CV.
                ncpus:            number of cpus to use for parallelization.
            Outputs:
                idx:              the selected variables
                criterionBestVal: the criterion value estimated for each idx by nfold-fold cv
        """
        # Get some information from the variables
        n = samples.shape[0]      # Number of samples
        if ncpus is None:
            ncpus=mp.cpu_count() # Get the number of core

        # Initialization
        nbSelectFeat     = 0                       # Initialization of the counter
        variables        = sp.arange(self.d)       # At step zero: d variables available
        idx              = []                      # and no selected variable
        criterionBestVal = []                      # list of the evolution the OA estimation
        idxBestSets      = []

        if maxvar==0.2:
            maxvar = sp.floor(self.d*maxvar) # Select at max maxvar % of the original number of variables

        # Start the forward search
        while(nbSelectFeat<maxvar) and (variables.size!=0):

            # Compute criterion function
            if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':
                # Parallelize cv
                pool = mp.Pool(processes=ncpus)
                processes =  [pool.apply_async(compute_metric_gmm, args=('forward',criterion,variables,model_pre_cv[k],samples[testInd,:],labels[testInd],idx)) for k, (trainInd,testInd) in enumerate(kfold)]
                pool.close()
                pool.join()

                # Compute mean criterion value over each processus
                criterionVal = sp.zeros(variables.size)
                for p in processes:
                    criterionVal += p.get()
                criterionVal /= len(kfold)
                del processes,pool

            elif criterion == 'JM':
                criterionVal = compute_JM('forward',variables,self,idx)

            elif criterion == 'divKL':
                criterionVal = compute_divKL('forward',variables,self,idx)


            # Select the variable that provides the highest criterion
            nbSelectFeat += 1
            bestVar = sp.argmax(criterionVal) # get the indice of the maximum of criterion values
            if nbSelectFeat <= len(criterionBestVal) and criterionVal[bestVar] < criterionBestVal[nbSelectFeat-1]:
                idx       = idxBestSets[nbSelectFeat-1][0]
                variables = idxBestSets[nbSelectFeat-1][1]
                # print sp.sort(idx)," jump"
            else:

                idx.append(variables[bestVar])
                # print sp.sort(idx)
                variables = sp.delete(variables,bestVar)  # remove the selected variables from the initial set
                if nbSelectFeat > len(criterionBestVal):
                    criterionBestVal.append(criterionVal[bestVar])   # save criterion value
                    idxBestSets.append((list(idx),variables))
                else:
                    criterionBestVal[nbSelectFeat-1] = criterionVal[bestVar]   # save criterion value
                    idxBestSets[nbSelectFeat-1] = (list(idx),variables)

                # print "add ",idx
                flagBacktrack = True

                while flagBacktrack and nbSelectFeat > 2:

                    # Compute criterion function
                    if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':
                        # Parallelize cv
                        pool = mp.Pool(processes=ncpus)
                        processes =  [pool.apply_async(compute_metric_gmm, args=('backward',criterion,sp.array(idx),model_pre_cv[k],samples[testInd,:],labels[testInd],idx)) for k, (trainInd,testInd) in enumerate(kfold)]
                        pool.close()
                        pool.join()

                        # Compute mean criterion value over each processus
                        criterionVal = sp.zeros(len(idx))
                        for p in processes:
                            criterionVal += p.get()
                        criterionVal /= len(kfold)
                        del processes,pool

                    elif criterion == 'JM':
                        criterionVal = compute_JM('backward',sp.array(idx),self,idx)

                    elif criterion == 'divKL':
                        criterionVal = compute_divKL('backward',sp.array(idx),self,idx)


                    bestVar = sp.argmax(criterionVal) # get the indice of the maximum of criterion values

                    if criterionVal[bestVar] > criterionBestVal[nbSelectFeat-2]:
                        nbSelectFeat -= 1
                        variables = sp.sort( sp.append(variables,idx[bestVar]) )
                        del idx[bestVar]
                        # print "del ", idx

                        criterionBestVal[nbSelectFeat-1] = criterionVal[bestVar]   # save criterion value
                        idxBestSets[nbSelectFeat-1] = (list(idx),variables)
                    else:
                        flagBacktrack = False

        ## Return the final value
        return idx,criterionBestVal

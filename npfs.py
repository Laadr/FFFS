# -*- coding: utf-8 -*-
'''
Created on 6 april 2016

@author: adlagrange
'''

import scipy as sp
from scipy import linalg
import multiprocessing as mp
import sklearn.cross_validation as cv

## Utilitary functions
def mylstsq(a, b, rcond):
    """
        Compute a safe/fast least square fitting.
        For that particular case, the number of unknown parameters is equal to the number of equations.
        However, a is a covariance matrix that might estimated with a number of samples less than the number of variables, leading to a badly conditionned covariance matrix.
        So, the rcond number is check: if it is ok, we use the fast linalg.solve function; otherwise, we use the slow, but safe, linalg.lstsq function.
        Inputs:
        a: a symmetric definite positive matrix d times d
        b: a d times n matrix
        Outputs:
        x: a d times n matrix
    """
    eps = sp.finfo(sp.float64).eps
    if rcond>eps: # If the condition number is not too bad try linear system
        try:
            x = linalg.solve(a,b)
        except linalg.LinAlgError: # If error, use least square estimations
            x = linalg.lstsq(a,b)[0]
    else:# Use least square estimate
        x = linalg.lstsq(a,b)[0]
    return x

def safe_logdet(cov):
    '''
    The function computes a secure version of the logdet of a covariance matrix and it returns the rcondition number of the matrix.
    Inputs:
        cov: a matrix
    Outputs:
        logdet: the 'safe' logdet of cov
        rcond:  the conditioning of cov
    '''
    eps = sp.finfo(sp.float64).eps
    e = linalg.eigvalsh(cov)
    if e.max()<eps:
        rcond = 0
    else:
        rcond = e.min()/e.max()
    e = sp.where(e<eps,eps,e)
    return sp.sum(sp.log(e)),rcond

def compute_loocv_gmm(var,model,samples,labels,idx,K_u,alpha,beta,log_prop_u, tau=None):
    """
        Function that computes the accuracies of the GMM model (updated with one sample out)  with variables: idx + var
        Inputs:
            model: the GMM model
            samples,labels: the training samples and the corresponding label
            idx: the pool of selected variables
            variable: the variable to be tested from the set of available variable
            K_u: the initial prediction values computed with all the samples
            alpha, beta and log_prop_u : constant that are computed outside of the loop to increased speed
        Outputs:
            loocv_temp : the loocv

        Used in GMM.forward_selection()
    """
    # Get information on samples
    n = samples.shape[0]

    # Prepare pool of variables
    id_t = list(idx)
    id_t.append(var)
    id_t.sort()

    # Compute decision fct with all the samples with id_t
    Kp = model.predict_gmm(samples,featIdx=id_t,tau=tau)[1]

    # Initialization of the temporary loo accuracy
    loocv_temp = 0.

    for j in xrange(n):
        c    = int(labels[j]-1)
        Kloo = Kp[j,:] + K_u  # Initialization of the decision rule for sample "j" #--- Change for only not c---#

        # Update of parameter of class c
        m            = (model.nbSpl[c]*model.mean[c,id_t] - samples[j,id_t])*alpha[c] # Update the mean value
        xb           = samples[j,id_t] - m # x centered
        cov_u        = model.cov[c,id_t,:][:,id_t] - sp.outer(xb,xb)*alpha[c]*beta[c] # Update the covariance matrix
        logdet,rcond = safe_logdet(cov_u)

        if tau == None:
            temp = mylstsq(cov_u,xb.T,rcond)
            # temp = sp.dot(linalg.inv(cov_u), xb.T)
        else:
            temp = linalg.lstsq( sp.dot(cov_u,cov_u) + tau * sp.ones(len(id_t)) , sp.dot(cov_u,xb.T))[0]
            # temp = sp.dot( linalg.inv( sp.dot(cov_u,cov_u) + tau * sp.ones(len(id_t)) ), sp.dot(cov_u,xb.T) )

        Kloo[c] = logdet - 2*log_prop_u[c] + sp.vdot(xb,temp) # Compute the new decision rule
        del cov_u,xb,m,c

        yloo = sp.argmin(Kloo)+1
        loocv_temp += float(yloo==labels[j]) # Check the correct/incorrect classification rule

    return loocv_temp/n                                           # Compute loocv for variable

def compute_metric_gmm(direction, criterion, variables, model_cv, samples, labels, idx, tau=None, decisionMethod='linsyst'):
    """
        Function that computes the accuracy of the model_cv using the variables : idx + one of variables
        Inputs:
            direction:      'backward' or 'forward'
            criterion:      criterion function use to discriminate variables
            variables:      the variable to add or delete to idx
            model_cv:       the model build with all the variables
            samples,labels: the samples/label for testing
            idx:            the pool of retained variables
            tau:            regularization parameter
            decisionMethod: method to compute main term of decision function 'linsyst' or 'inv'. Default: 'linsyst'
        Output:
            metric: the estimated metric

        Used in GMM.forward_selection(), GMM.backward_selection()
    """
    metric     = sp.zeros(variables.size)
    confMatrix = ConfusionMatrix()
    idx        = sp.sort(idx)

    # Compute inv of covariance matrix
    if decisionMethod == 'inv':
        if len(idx)==0:
            invCov = None
            logdet = None
        else:
            invCov     = sp.empty((model_cv.C,len(idx),len(idx)))
            logdet     = sp.empty((model_cv.C))

            for c in xrange(model_cv.C):
                vp,Q,rcond    = model_cv.decomposition(model_cv.cov[c,idx,:][:,idx],tau)
                invCov[c,:,:] = sp.dot(Q,((1/vp)*Q).T)
                logdet[c]     = sp.sum(sp.log(vp))

    for i,var in enumerate(variables):
        if decisionMethod=='inv':
            predLabels = model_cv.predict_gmm_maj(direction,samples,invCov,logdet,(i,var),featIdx=idx,tau=tau)[0]
        else:
            if direction=='forward':
                id_t = list(idx)
                id_t.append(var)
                id_t.sort()
            elif direction=='backward':
                mask    = sp.ones(len(variables), dtype=bool)
                mask[i] = False
                id_t    = list(idx[mask])
            predLabels = model_cv.predict_gmm(samples,featIdx=id_t,tau=tau,decisionMethod=decisionMethod)[0] # Use the marginalization properties to update the model for each tuple of variables
            del id_t

        confMatrix.compute_confusion_matrix(predLabels,labels)
        if criterion=='accuracy':
            metric[i] = confMatrix.OA
        elif criterion=='F1Mean':
            metric[i] = confMatrix.F1Mean
        elif criterion=='kappa':
            metric[i] = confMatrix.Kappa

    return metric

def compute_JM(direction, variables, model, idx, tau=None):
    """
        Function that computes the Jeffries–Matusita distance of the model_cv using the variables : idx + one of variables
        Inputs:
            variables: the variable to add to idx
            model_cv:  the model build with all the variables
            idx:       the pool of retained variables
        Output:
            JM: the estimated Jeffries–Matusita distance

        Used in GMM.forward_selection()
    """
    # # Initialization
    # JM = sp.zeros(variables.size)

    # # For all variables
    # for k,var in enumerate(variables):
    #     id_t = list(idx)
    #     id_t.append(var)
    #     id_t.sort()

    #     for i in xrange(model.C):
    #         for j in xrange(i+1,model.C):
    #             md  = (model.mean[i,id_t]-model.mean[j,id_t])
    #             cs  = (model.cov[i,id_t,:][:,id_t]+model.cov[j,id_t,:][:,id_t])/2
    #             di  = linalg.det(model.cov[i,id_t,:][:,id_t])
    #             dj  = linalg.det(model.cov[j,id_t,:][:,id_t])
    #             dij = linalg.det(cs)
    #             bij = sp.dot(md,linalg.solve(cs,md))/8 + 0.5*sp.log(0.5*dij/sp.sqrt(di*dj))
    #             JM[k] += sp.sqrt(2*(1-sp.exp(-bij)))*model.prop[i]*model.prop[j]

    # Initialization
    JM = sp.zeros(variables.size)
    d  = sp.zeros((model.C,variables.size))

    # Cast and sort index of selected variables
    idx = list(idx)
    idx.sort()


    if tau==None:
        tau=0

    # Compute all possible update of det cov(idx)
    if len(idx)==0:
        for c in xrange(model.C):
            for k,var in enumerate(variables):
                d[c,k] = model.cov[c,var,var]
    else:
        for c in xrange(model.C):
            vp,Q,rcond = model.decomposition(model.cov[c,idx,:][:,idx],tau)
            det = sp.prod(vp)
            invCov = sp.dot(Q,((1/vp)*Q).T)
            for k,var in enumerate(variables):
                if direction=='forward':
                    maj_cst = model.cov[c,var,var] + tau - sp.dot(model.cov[c,var,:][idx], sp.dot(invCov,model.cov[c,var,:][idx].T) )
                elif direction=='backward':
                    maj_cst     = 1/float( invCov[k,k] )
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
                vp,Q,rcond = model.decomposition(cs,tau)
                invCov = sp.dot(Q,((1/vp)*Q).T)
                det = sp.prod(vp)

                for k,var in enumerate(variables):
                    id_t = list(idx)
                    id_t.append(var)
                    id_t.sort()

                    md      = (model.mean[i,idx]-model.mean[j,idx])

                    if direction=='forward':
                        c1 = (model.cov[i,var,var]+model.cov[j,var,var])/2
                        c2 = (model.cov[i,var,:][idx]+model.cov[j,var,:][idx])/2
                        maj_cst = c1 + tau - sp.dot(c2, sp.dot(invCov,c2.T) )
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

## Confusion matrix

class ConfusionMatrix(object):
    def __init__(self):
        self.confusion_matrix = None
        self.OA               = None
        self.Kappa            = None
        self.F1Mean           = None

    def compute_confusion_matrix(self,yp,yr):
        """
            Compute the confusion matrix
            Inputs:
                yp: predicted labels
                yr: reference labels
        """
        # Initialization
        n = yp.size
        C = int(yr.max())
        self.confusion_matrix = sp.zeros((C,C))

        # Compute confusion matrix
        for i in xrange(n):
            self.confusion_matrix[yp[i].astype(int)-1, yr[i].astype(int)-1] += 1

        # Compute overall accuracy
        self.OA = sp.sum(sp.diag(self.confusion_matrix))/n

        # Compute Kappa
        nl = sp.sum(self.confusion_matrix,axis=1)
        nc = sp.sum(self.confusion_matrix,axis=0)

        self.Kappa = ((n**2)*self.OA - sp.sum(nc*nl))/(n**2-sp.sum(nc*nl))

        # Compute F1 Mean
        self.F1Mean = sp.sum(2*sp.diag(self.confusion_matrix) / (nl + nc))/n

## Gaussian Mixture Model

class GMM(object):

    def __init__(self, d=0, C=0):
        self.nbSpl   = sp.empty((C,1)) # array of number of samples in each class
        self.prop    = sp.empty((C,1)) # array of proportion in training set
        self.mean    = sp.empty((C,d)) # array of means
        self.cov     = sp.empty((C,d,d)) # array of covariance matrices
        self.C       = C            # number of class
        self.d       = d            # number of features

    def decomposition(self, M, tau=None):
        """
            Compute the decompostion of symmetric matrix
            Inputs:
                M:   matrix to decompose
                tau: regularisation term added to eigenvalues
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

        if tau == None:
            vp = sp.where(vp<eps,eps,vp)
        else:
            vp += tau

        return vp,Q,rcond

    def learn_gmm(self, samples, labels):
        """
            Method that learns the GMM from training samples and store the mean, covariance and proportion of each class in class members.
            Input:
                samples: the training samples
                labels:  the labels
        """
        # Get information from the data
        self.C = int(labels.max(0)) # Number of classes
        self.d = samples.shape[1]   # Number of variables

        # Initialization
        self.nbSpl = sp.empty((self.C,1))   # Vector of number of samples for each class
        self.prop  = sp.empty((self.C,1))   # Vector of proportion
        self.mean  = sp.empty((self.C,self.d))   # Vector of means
        self.cov   = sp.empty((self.C,self.d,self.d)) # Matrix of covariance

        # Learn the parameter of the model for each class
        for i in xrange(self.C):
            # Get index of class i+1 samples
            j = sp.where(labels==(i+1))[0]

            # Update GMM
            self.nbSpl[i]   = float(j.size)
            self.mean[i,:]  = sp.mean(samples[j,:],axis=0)
            self.cov[i,:,:] = sp.cov(samples[j,:],rowvar=0)  # Normalize by ni to be consistent with the update formulae

        self.prop = self.nbSpl/samples.shape[0]

    def predict_gmm(self, testSamples, featIdx=None, tau=None, decisionMethod='linsyst'):
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
        if featIdx == None:
            featIdx = range(testSamples.shape[1])

        # Start the prediction for each class
        for c in xrange(self.C):
            testSamples_c = testSamples[:,featIdx] - self.mean[c,featIdx]
            vp,Q,rcond    = self.decomposition(self.cov[c,featIdx,:][:,featIdx],tau)
            logdet        = sp.sum(sp.log(vp))
            cst           = logdet - 2*sp.log(self.prop[c]) # Pre compute the constant term

            if decisionMethod=='inv':
                temp = sp.dot( np.dot(Q,((1/vp)*Q).T) , testSamples_c.T).T
            else:
                temp = mylstsq(self.cov[c,featIdx,:][:,featIdx],testSamples_c.T,rcond).T

            scores[:,c] = sp.sum(testSamples_c*temp,axis=1) + cst

            del temp,testSamples_c

        # Assign the label to the minimum value of scores
        predLabels = sp.argmin(scores,1)+1
        return predLabels,scores


class GMMFeaturesSelection(GMM):

    def __init__(self, d=0, C=0):
        super(GMMFeaturesSelection, self).__init__(d,C)

    def predict_gmm_maj(self, direction, testSamples, invCov, logdet, newFeat, featIdx=None, tau=None):
        """
            Function that predict the label for testSamples using the learned model (with an update method of the inverte covariance matrix)
            Inputs:
                direction:   'backward' or 'forward'
                testSamples: the samples to be classified
                invCov:      inverte of covariance matrix of already selected features
                newFeat:     couple (index,feature) to add or delete
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
        if featIdx == None:
            featIdx = range(testSamples.shape[1])

        # New set of features
        if direction=='forward':
            featIdx  = list(featIdx)
            featIdx.sort()
            id_t = list(featIdx)
            id_t.append(newFeat[1])
            id_t.sort()
        elif direction=='backward':
            id_t = list(featIdx)
            id_t.sort()

        if tau==None:
            tau=0

        # Start the prediction for each class
        for c in xrange(self.C):
            testSamples_c = testSamples[:,id_t] - self.mean[c,id_t]

            if logdet==None and invCov==None:
                logdet_maj = sp.sum(sp.log(self.cov[c,newFeat[1],newFeat[1]] + tau))

                tmp = float(self.cov[c,id_t,id_t])
                temp = sp.dot([[tmp/(tmp**2 + tau)]], testSamples_c.T).T
                scores[:,c] = sp.sum(testSamples_c*temp,axis=1) + logdet_maj - 2*sp.log(self.prop[c])
            else:
                if direction=='forward':
                    d_feat     = self.cov[c,newFeat[1],newFeat[1]] + tau - sp.dot(self.cov[c,newFeat[1],:][featIdx], sp.dot(invCov[c,:,:][:,:],self.cov[c,newFeat[1],:][featIdx].T) )
                    logdet_maj = sp.log(d_feat) + logdet[c]

                    ind           = id_t.index(newFeat[1])
                    row_feat      = -1/float(d_feat) * sp.dot(self.cov[c,:,newFeat[1]][featIdx],invCov[c,:,:][:,:])
                    row_feat      = sp.insert( row_feat, ind, 1/float(d_feat) )
                    cst_feat      = d_feat * (sp.dot(row_feat,testSamples_c.T)**2)
                    testSamples_c = testSamples[:,featIdx] - self.mean[c,featIdx]

                elif direction=='backward':
                    d_feat     = 1/float( invCov[c,newFeat[0],newFeat[0]] )
                    logdet_maj = sp.log(d_feat) - logdet[c]

                    row_feat   = invCov[c,newFeat[0],:]
                    cst_feat   = - d_feat * (sp.dot(row_feat,testSamples_c.T)**2)

                cst = logdet_maj - 2*sp.log(self.prop[c]) # Pre compute the constant term

                temp = sp.dot(invCov[c,:,:][:,:], testSamples_c.T).T

                scores[:,c] = sp.sum(testSamples_c*temp,axis=1) + cst_feat + cst

            del temp,testSamples_c

        # Assign the label to the minimum value of scores
        predLabels = sp.argmin(scores,1)+1
        return predLabels,scores

    def selection_cv(self, direction, samples, labels, criterion='accuracy', stopMethod='maxVar', delta=0.1, maxvar=0.2, nfold=5, balanced=True, tau=None, ncpus=None, decisionMethod='linsyst'):
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
                tau:             regularization term added to eigenvalues. Default: None
                ncpus:           number of cpus to use for parallelization. Default: all
                decisionMethod:  'linsyst' to use least quare to compute decision, 'inv' to use matrix inv computed by matrix diaginalization to compute decision. Default: 'linsyst'

            Outputs:
                idx:              the selected variables
                criterionBestVal: the criterion value estimated for each idx by nfold-fold cv
        """
        # Get some information from the variables
        n = samples.shape[0]      # Number of samples

        # Creation of folds
        if balanced:
            kfold = cv.StratifiedKFold(labels.ravel(),n_folds=nfold,shuffle=True) # kfold is an iterator
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
                nk_c = float(classInd.size)
                mean_k = sp.mean(testSamples[classInd,:],axis=0)
                cov_k = sp.cov(testSamples[classInd,:],rowvar=0)

                model_pre_cv[k].nbSpl[c]  = self.nbSpl[c] - nk_c
                model_pre_cv[k].mean[c,:] = (self.nbSpl[c]*self.mean[c,:]-nk_c*mean_k)/(self.nbSpl[c]-nk_c)
                model_pre_cv[k].cov[c,:]  = ((self.nbSpl[c]-1)*self.cov[c,:,:] - nk_c*cov_k - nk_c*self.nbSpl[c]/model_pre_cv[k].nbSpl[c]*sp.outer(self.mean[c,:]-mean_k,self.mean[c,:]-mean_k))/model_pre_cv[k].nbSpl[c]
                del classInd,nk_c,mean_k,cov_k

            # Update proportion
            model_pre_cv[k].prop = model_pre_cv[k].nbSpl/(n-nk)

            del testSamples,testLabels,nk

        if direction == 'forward':
            idx,criterionBestVal = self.forward_selection(samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, tau, ncpus, decisionMethod)
        elif direction == 'backward':
            idx,criterionBestVal = self.backward_selection(samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, tau, ncpus, decisionMethod)

        return idx,criterionBestVal

    def forward_selection(self, samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, tau, ncpus, decisionMethod):
        """
            Function that selects the most discriminative variables according to a forward search
            Inputs:
                samples, labels: the training samples and their labels
                criterion:       the criterion function to use for selection (accuracy, kappa, F1Mean, JM).
                stopMethod:      the stopping criterion. It can be either 'maxVar' to continue until maxvar variables are selected or either 'variation' to continue until variation of criterion function are less than delta.
                delta:           the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta.
                maxvar:          maximum number of extracted variables.
                kfold:           k-folds for the cross-validation.
                model_pre_cv:    GMM models for each CV.
                tau:             regularization term added to eigenvalues. Default: None
                ncpus:           number of cpus to use for parallelization.
                decisionMethod:  'linsyst' to use least quare to compute decision, 'inv' to use matrix inv computed by matrix diaginalization to compute decision. Default: 'linsyst'

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

            # Parallelize cv
            pool = mp.Pool(processes=ncpus)
            if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':
                processes =  [pool.apply_async(compute_metric_gmm, args=('forward',criterion,variables,model_pre_cv[k],samples[testInd,:],labels[testInd],idx,tau,decisionMethod)) for k, (trainInd,testInd) in enumerate(kfold)]
            elif criterion == 'JM':
                processes =  [pool.apply_async(compute_JM, args=('forward',variables,model_pre_cv[k],idx,tau)) for k in xrange(len(kfold))]
            pool.close()
            pool.join()

            # Compute mean criterion value over each processus
            criterionVal = sp.zeros(variables.size)
            for p in processes:
                criterionVal += p.get()
            criterionVal /= len(kfold)
            del processes,pool

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

    def backward_selection(self, samples, labels, criterion, stopMethod, delta, maxvar, kfold, model_pre_cv, tau, ncpus, decisionMethod):
        """
            Function that selects the most discriminative variables according to a backward search
            Inputs:
                samples, labels:  the training samples and their labels
                criterion:        the criterion function to use for selection (accuracy or JM).
                stopMethod:       the stopping criterion. It can be either 'maxVar' to continue until maxvar variables are selected or either 'variation' to continue until variation of criterion function are less than delta.
                delta :           the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta.
                maxvar:           maximum number of extracted variables.
                kfold:            k-folds for the cross-validation.
                model_pre_cv:     GMM models for each CV.
                tau:              regularization term added to eigenvalues. Default: None
                ncpus:            number of cpus to use for parallelization.
                decisionMethod:   'linsyst' to use least quare to compute decision, 'inv' to use matrix inv computed by matrix diaginalization to compute decision. Default: 'linsyst'
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

            # Parallelize cv
            pool = mp.Pool(processes=ncpus)
            if criterion == 'accuracy' or criterion == 'F1Mean' or criterion == 'kappa':
                processes =  [pool.apply_async(compute_metric_gmm, args=('backward',criterion,idx,model_pre_cv[k],samples[testInd,:],labels[testInd],idx,tau,decisionMethod)) for k, (trainInd,testInd) in enumerate(kfold)]
            elif criterion == 'JM':
                processes =  [pool.apply_async(compute_JM, args=('backward',idx,model_pre_cv[k],idx,tau)) for k in xrange(len(kfold))]
            pool.close()
            pool.join()

            # Compute mean criterion value over each processus
            criterionVal = sp.zeros(idx.size)
            for p in processes:
                criterionVal += p.get()
            criterionVal /= len(kfold)
            del processes,pool

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

    # def forward_selection_loo(self, samples, labels, stopMethod='maxVar', delta=0.1, maxvar=0.2, ncpus=None, tau=None):
    #     """
    #         Function that selects the most discriminative variables according to a forward search using Leave-One-Out method
    #         Inputs:
    #             samples, labels:  the training samples and their labels
    #             stopMethod: the stopping criterion. It can be either 'maxVar' to continue until maxvar % of the variables are selected or either 'variation' to continue until variation of criterion function are less than delta. Default: 'maxVar'
    #             delta :  the minimal improvement in percentage when a variable is added to the pool, the algorithm stops if the improvement is lower than delta. Default value 0.1%
    #             maxvar: maximum number of extracted variables. Default value: 20% of the original number
    #             ncpus: number of cpus to use for parallelization. Default: all
    #             tau: regularization parameter. Default: None

    #         Outputs:
    #             idx: the selected variables
    #             criterionBestVal: the criterion value estimated for each idx by loocv
    #     """
    #     # Get some information from the variables
    #     n = samples.shape[0]      # Number of samples
    #     if ncpus is None:
    #         ncpus=mp.cpu_count() # Get the number of core

    #     # Initialization
    #     nbSelectFeat     = 0                       # Initialization of the counter
    #     variables        = sp.arange(self.d)      # At step zero: d variables available
    #     idx              = []                      # and no selected variable
    #     criterionBestVal = []                      # list of the evolution the OA estimation
    #     maxvar           = sp.floor(self.d*maxvar) # Select at max maxvar % of the original number of variables

    #     # Precompute the proportion and some others constants
    #     log_prop_u = [sp.log((n*self.prop[c]-1.0)/(n-1.0)) for c in range(self.C)]
    #     K_u        = 2*sp.log((n-1.0)/n)
    #     beta       = [self.nbSpl[c]/(self.nbSpl[c]-1.0) for c in range(self.C)]# Constant for the rank one downdate
    #     alpha      = [1/(self.nbSpl[c]-1) for c in range(self.C)]

    #     # Start the forward search
    #     while(nbSelectFeat<maxvar) and (variables.size!=0):
    #         # Parallelize for each variable
    #         pool = mp.Pool(processes=ncpus)
    #         processes =  [pool.apply_async(compute_loocv_gmm,args=(var,self,samples,labels,idx,K_u,alpha,beta,log_prop_u,tau)) for var in variables]
    #         pool.close()
    #         pool.join()

    #         # Compute mean criterion value over each processus
    #         criterionVal = sp.zeros(variables.size)
    #         for i,p in enumerate(processes):
    #             criterionVal[i] += p.get()
    #         del processes,pool

    #         # Select the variable that provides the highest loocv
    #         bestVar = sp.argmax(criterionVal)                # get the indice of the maximum of criterion values
    #         criterionBestVal.append(criterionVal[bestVar])         # save criterion value

    #         if nbSelectFeat==0:
    #             idx.append(variables[bestVar])           # add the selected variables to the pool
    #             variables = sp.delete(variables,bestVar)  # remove the selected variables from the initial set
    #         elif (stopMethod=='variation') and (((criterionBestVal[nbSelectFeat]-criterionBestVal[nbSelectFeat-1])/criterionBestVal[nbSelectFeat-1]*100) < delta):
    #             criterionBestVal.pop()
    #             break
    #         else:
    #             idx.append(variables[bestVar])
    #             variables=sp.delete(variables,bestVar)
    #         nbSelectFeat += 1

    #     ## Return the final value
    #     return idx,criterionBestVal

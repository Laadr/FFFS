# -*- coding: utf-8 -*-
import scipy as sp
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import npfs as npfs

data_name  = 'aisa'
ntrial = 20

Nts        = [(50,'equalized'), (100,'equalized'), (200,'equalized'), (400,'equalized'), (0.025,'stratified'), (0.05,'stratified')]#, (0.3,True)] # Nb of samples per class in training set
criterions = ['accuracy', 'kappa', 'F1Mean', 'JM', 'divKL']

mean_forward_time = sp.zeros( (len(Nts)) )
std_forward_time  = sp.zeros( (len(Nts)) )
mean_SFFS_time    = sp.zeros( (len(Nts)) )
std_SFFS_time     = sp.zeros( (len(Nts)) )

plt.figure(1,figsize=(15,15))
for criterion in criterions:
    plt.clf()
    i=0
    for Nt,stratification in Nts:

        if stratification=='stratified':
            filename  = data_name+'_forward_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)+'_'+criterion
            filename2 = data_name+'_SFFS_spls_'+str(Nt)+'_stratified_trials_'+str(ntrial)+'_'+criterion
            labelExtension = " sampling stratified with "+str(Nt*100)+"%"
        else:
            filename  = data_name+'_forward_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)+'_'+criterion
            filename2 = data_name+'_SFFS_spls_'+str(Nt)+'_equalized_trials_'+str(ntrial)+'_'+criterion
            labelExtension = " sampling equalized with "+str(Nt)+" samples"

        f = open('Results/'+filename + '.pckl')
        res1 = pickle.load(f)
        f.close()
        f = open('Results/'+filename2 + '.pckl')
        res2 = pickle.load(f)
        f.close()

        # p = 0
        # for i in xrange(len(res1)):
        #     p += sp.stats.ranksums(res1[1][1],res2[1][1])[1]
        # print "Wilcoxon rank-sum statistic (p-value): ", p
        maxVar = len(res1[0][1])
        plt.subplot(311)
        x = range(5,5+maxVar)
        dataTmp = sp.asarray([res1[j][1].ravel() for j in xrange(len(res1))])
        y = sp.asarray([sp.mean(dataTmp[:,k])/100 for k in xrange(len(x))])
        error = sp.asarray([sp.std(dataTmp[:,k])/100 for k in xrange(len(x))])
        plt.plot(x, y, label='Forward '+labelExtension)
        # plt.fill_between(x, y-error, y+error, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        dataTmp = sp.asarray([res2[j][1].ravel() for j in xrange(len(res1))])
        y = sp.asarray([sp.mean(dataTmp[:,k])/100 for k in xrange(len(x))])
        error = sp.asarray([sp.std(dataTmp[:,k])/100 for k in xrange(len(x))])
        plt.plot(x, y, label='SFFS '+labelExtension)
        # plt.fill_between(x, y-error, y+error, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)

        plt.subplot(312)
        dataTmp = sp.asarray([res1[j][2].ravel() for j in xrange(len(res1))])
        plt.plot(range(5,maxVar+5), [sp.mean(dataTmp[:,k]) for k in xrange(len(x))], label='Forward '+labelExtension)
        dataTmp = sp.asarray([res2[j][2].ravel() for j in xrange(len(res1))])
        plt.plot(range(5,maxVar+5), [sp.mean(dataTmp[:,k]) for k in xrange(len(x))], label='SFFS '+labelExtension)

        plt.subplot(313)
        dataTmp = sp.asarray([res1[j][3].ravel() for j in xrange(len(res1))])
        plt.plot(range(5,maxVar+5), [sp.mean(dataTmp[:,k]) for k in xrange(len(x))], label='Forward '+labelExtension)
        dataTmp = sp.asarray([res2[j][3].ravel() for j in xrange(len(res1))])
        plt.plot(range(5,maxVar+5), [sp.mean(dataTmp[:,k]) for k in xrange(len(x))], label='SFFS '+labelExtension)

        dataTmp = sp.asarray([res1[j][4] for j in xrange(len(res1))])
        mean_forward_time[i] = sp.mean(dataTmp)
        std_forward_time[i]  = sp.std(dataTmp)
        dataTmp = sp.asarray([res2[j][4] for j in xrange(len(res1))])
        mean_SFFS_time[i]    = sp.mean(dataTmp)
        std_SFFS_time[i]     = sp.std(dataTmp)

        i+=1

    plt.subplot(311)
    # plt.ylim((0.5,1))
    plt.legend(loc='center',bbox_to_anchor=(1.25, 0.7))
    plt.xlabel('Features nb')
    plt.ylabel('Overall Accuracy')
    plt.title(r'Classification score')


    plt.subplot(312)
    # plt.ylim((0.5,1))
    plt.legend(loc='center',bbox_to_anchor=(1.25, 0.7))
    plt.xlabel('Features nb')
    plt.ylabel('Kappa')
    plt.title(r'Classification score')


    plt.subplot(313)
    # plt.ylim((0.5,1))
    plt.legend(loc='center',bbox_to_anchor=(1.25, 0.7))
    plt.xlabel('Features nb')
    plt.ylabel('F1-score Mean')
    plt.title(r'Classification score')

    # plt.tight_layout()
    plt.savefig('Fig/'+data_name+'_trials_'+str(ntrial)+'_'+criterion, bbox_inches='tight')

    plt.clf()
    index = sp.arange(len(Nts))
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, mean_forward_time, bar_width, alpha=opacity, color='b', yerr=std_forward_time, error_kw=error_config, label='Forward selection')
    rects2 = plt.bar(index + bar_width, mean_SFFS_time, bar_width, alpha=opacity, color='r', yerr=std_SFFS_time, error_kw=error_config, label='SFFS selection')

    plt.xlabel('Number of samples')
    plt.ylabel('Computational time (s)')
    plt.xticks(index + bar_width, tuple(Nts))
    plt.legend(loc='upper left')
    plt.title(r'Computational time with '+str(len(res1[0][0]))+' features')

    plt.tight_layout()
    plt.savefig('Fig/time_'+data_name+'_trials_'+str(ntrial)+'_'+criterion)

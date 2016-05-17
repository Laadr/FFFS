# -*- coding: utf-8 -*-
import scipy as sp
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import npfs as npfs

data_name  = 'aisa'
ntrial = 2

Nts        = [(50,'equalized'), (100,'equalized'), (200,'equalized'), (0.1,'stratified')]#, (0.3,True)] # Nb of samples per class in training set
criterions = ['accuracy', 'kappa', 'F1Mean', 'JM', 'divKL']

mean_forward_start = sp.zeros( (len(Nts)) )
std_forward_start  = sp.zeros( (len(Nts)) )
mean_SFFS_start    = sp.zeros( (len(Nts)) )
std_SFFS_start     = sp.zeros( (len(Nts)) )

mean_forward_end = sp.zeros( (len(Nts)) )
std_forward_end  = sp.zeros( (len(Nts)) )
mean_SFFS_end    = sp.zeros( (len(Nts)) )
std_SFFS_end     = sp.zeros( (len(Nts)) )

plt.figure(1)
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
            labelExtension = " sampling equalized with "+str(Nt)+"samples"

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

        plt.subplot(311)
        x = range(len(res1[0][0][0]),len(res1[-1][0][0])+1)
        y = sp.asarray([sp.mean(res1[k][1])/100 for k in xrange(len(res1))])
        error = sp.asarray([sp.std(res1[k][1])/100 for k in xrange(len(res1))])
        plt.plot(x, y, label='Forward '+labelExtension)
        # plt.fill_between(x, y-error, y+error, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)
        y = sp.asarray([sp.mean(res2[k][1])/100 for k in xrange(len(res2))])
        error = sp.asarray([sp.std(res2[k][1])/100 for k in xrange(len(res2))])
        plt.plot(x, y, label='SFFS '+labelExtension)
        # plt.fill_between(x, y-error, y+error, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', antialiased=True)

        plt.subplot(312)
        plt.plot(range(len(res1[0][0][0]),len(res1[-1][0][0])+1), [sp.mean(res1[k][2]) for k in xrange(len(res1))], label='Forward '+labelExtension)
        plt.plot(range(len(res2[0][0][0]),len(res2[-1][0][0])+1), [sp.mean(res2[k][2]) for k in xrange(len(res2))], label='SFFS '+labelExtension)

        plt.subplot(313)
        plt.plot(range(len(res1[0][0][0]),len(res1[-1][0][0])+1), [sp.mean(res1[k][3]) for k in xrange(len(res1))], label='Forward '+labelExtension)
        plt.plot(range(len(res2[0][0][0]),len(res2[-1][0][0])+1), [sp.mean(res2[k][3]) for k in xrange(len(res2))], label='SFFS '+labelExtension)

        mean_forward_start[i] = sp.mean(res1[0][4])
        std_forward_start[i]  = sp.std(res1[0][4])
        mean_SFFS_start[i]    = sp.mean(res2[0][4])
        std_SFFS_start[i]     = sp.std(res2[0][4])

        mean_forward_end[i] = sp.mean(res1[-1][4])
        std_forward_end[i]  = sp.std(res1[-1][4])
        mean_SFFS_end[i]    = sp.mean(res2[-1][4])
        std_SFFS_end[i]     = sp.std(res2[-1][4])

        i+=1

    plt.subplot(311)
    # plt.ylim((0.5,1))
    plt.legend(loc='upper left')
    plt.xlabel('Features nb')
    plt.ylabel('Overall Accuracy')
    plt.title(r'Classification score')


    plt.subplot(312)
    # plt.ylim((0.5,1))
    plt.legend(loc='upper left')
    plt.xlabel('Features nb')
    plt.ylabel('Kappa')
    plt.title(r'Classification score')


    plt.subplot(313)
    # plt.ylim((0.5,1))
    plt.legend(loc='upper left')
    plt.xlabel('Features nb')
    plt.ylabel('F1-score Mean')
    plt.title(r'Classification score')

    plt.tight_layout()
    plt.savefig('Fig/'+data_name+'_trials_'+str(ntrial)+'_'+criterion)

    plt.clf()
    index = sp.arange(len(Nts))
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    plt.subplot(211)

    rects1 = plt.bar(index, mean_forward_start, bar_width, alpha=opacity, color='b', yerr=std_forward_start, error_kw=error_config, label='Forward selection')
    rects2 = plt.bar(index + bar_width, mean_SFFS_start, bar_width, alpha=opacity, color='r', yerr=std_SFFS_start, error_kw=error_config, label='SFFS selection')

    plt.xlabel('Criterion')
    plt.ylabel('Computational time (s)')
    plt.xticks(index + bar_width, tuple(Nts))
    plt.legend()
    plt.title(r'Computational time for '+str(len(res1[0][0][0]))+' features')


    plt.subplot(212)

    rects1 = plt.bar(index, mean_forward_end, bar_width, alpha=opacity, color='b', yerr=std_forward_end, error_kw=error_config, label='Forward selection')
    rects2 = plt.bar(index + bar_width, mean_SFFS_end, bar_width, alpha=opacity, color='r', yerr=std_SFFS_end, error_kw=error_config, label='SFFS selection')

    plt.xlabel('Criterion')
    plt.ylabel('Computational time (s)')
    plt.xticks(index + bar_width, tuple(Nts))
    plt.legend()
    plt.title(r'Computational time for '+str(len(res1[-1][0][0]))+' features')


    plt.tight_layout()
    plt.savefig('Fig/time_'+data_name+'_trials_'+str(ntrial)+'_'+criterion)

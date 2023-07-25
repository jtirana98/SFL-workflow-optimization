import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 15})


def label_diff(i,j,text,X,Y, ii):
    x = (X[i]+X[j])/2
    y = 1.01*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':30,'shrinkB':30,'linewidth':1}
    axs[ii].annotate(text, xy=(X[i]+0.2,y+12), zorder=10)
    axs[ii].annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

fig, axs = plt.subplots(1,2)
fig.set_size_inches(6, 3)
labels = ('1', '2', '5', '10', '20', '25')
menStd     = ('2', '3', '4', '5')
width = 0.7
x = np.arange(len(labels))  # the label locations
ind  = np.arange(2)

resnet_diminish = (190.5,99.79,78.27,59.56,50.20,48.64)
vgg_diminish = (257.3, 136.2,91.69,53.52,39.84,37.98)

bar_kwargs = {'width':width,'color':(0.2, 0.4, 0.6, 0.6),'linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':None,'linewidth':2,'ecolor':'k'}   #for matplotlib >= v1.4 use 'fmt':'none' instead


axs[0].bar(x, resnet_diminish, **bar_kwargs)

#axs[0].errs = plt.errorbar(x, resnet_diminish, yerr=menStd, **err_kwargs)

dimish = ['47.6\%', '21.5\%', '23.9\%', '15.7\%', '3.1\%']
for j in range(1, len(x)):
    #axs[0].axvline(x=j, linewidth=1, color='black', ymin=0, ymax=0.5)
    axs[0].plot([j, j], [resnet_diminish[j], resnet_diminish[j-1]], color='black', linestyle = 'dashed', linewidth=1)
    label_diff(j-1,j,dimish[j-1],x,resnet_diminish, 0)





axs[1].bar(x, vgg_diminish, **bar_kwargs)
dimish = ['47\%', '32.7\%', '41.2\%', '25.5\%', '4.67\%']
for j in range(1, len(x)):
    #axs[0].axvline(x=j, linewidth=1, color='black', ymin=0, ymax=0.5)
    axs[1].plot([j, j], [vgg_diminish[j], vgg_diminish[j-1]], color='black', linestyle = 'dashed', linewidth=1)
    label_diff(j-1,j,dimish[j-1],x,vgg_diminish, 1)


# Add some text for labels, title and custom x-axis tick labels, etc.

axs[0].plot([0, 0], [800, 800], color='black', linewidth=1.5, label='ralative gain (\%)')
axs[0].legend(bbox_to_anchor=(1.5, 1.2), ncol=1)
#for ax in axs.flat:
axs[0].set_ylabel('batch makespan (sec)', fontsize=16)


fig.suptitle('number of compute nodes', y=0.02,fontsize=16)
    
    

for ax in axs:
    ax.set_xticks(x, labels)
    ax.set_ylim(30, 300)
    ax.grid(axis = "y") #for grid

plt.savefig("diminish.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()

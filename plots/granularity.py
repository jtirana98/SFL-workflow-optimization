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
    #axs[ii].annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)

props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':30,'shrinkB':30,'linewidth':1}

fig, axs = plt.subplots(1,2)
fig.set_size_inches(10, 3)
labels = ('(50,5)', '(50,10)')
ind  = np.arange(2)

resnet_gran = {
    '$|S_t|$ = 200': (64.2, 50),
    '$|S_t|$ = 150': (60.9, 49),
    '$|S_t|$ = 50': ( 53.7, 49.25),
}

vgg_gran = {
    '$|S_t|$ = 200': (54,  46.2),
    '$|S_t|$ = 150': (52.5, 45 ),
    '$|S_t|$ = 50': (49.95, 40.35),
}


x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0
iter = 0
hatches = ['///', 'o']
for attribute, measurement in resnet_gran.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[0].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        rects = axs[0].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
    #axs[0].bar_label(rects, padding=3) # to add value top on bars
    multiplier += 1
    iter += 1

speedup = ['x3.2', 'x4.3', 'x3.3', 'x4.2']
resnet_150 = resnet_gran['$|S_t|$ = 150']
resnet_200 = resnet_gran['$|S_t|$ = 200']


axs[0].annotate(speedup[0], xy=(0.3,resnet_150[0]), zorder=10,fontsize=20)
axs[0].annotate(speedup[1], xy=(0,resnet_200[0]), zorder=10,fontsize=20)
axs[0].annotate(speedup[2], xy=(1+0.3,resnet_150[1]), zorder=10,fontsize=20)
axs[0].annotate(speedup[3], xy=(1,resnet_200[1]+2), zorder=10,fontsize=20)


multiplier = 0
iter = 0
for attribute, measurement in vgg_gran.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[1].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        rects = axs[1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
    #axs[1].bar_label(rects, padding=3) # to add value top on bars
    multiplier += 1
    iter += 1

speedup = ['x3.8', 'x4.9', 'x3.4', 'x4.8']
vgg_150 = vgg_gran['$|S_t|$ = 150']
vgg_200 = vgg_gran['$|S_t|$ = 200']
axs[1].annotate(speedup[0], xy=(0,vgg_150[0]+3), zorder=10,fontsize=20)
axs[1].annotate(speedup[1], xy=(0.3,vgg_200[0]-1), zorder=10,fontsize=20)
axs[1].annotate(speedup[2], xy=(1+0.3,vgg_150[1]), zorder=10,fontsize=20)
axs[1].annotate(speedup[3], xy=(1,vgg_200[1]+2), zorder=10,fontsize=20)


# Add some text for labels, title and custom x-axis tick labels, etc.
#for ax in axs.flat:
axs[0].set_ylabel('batch makespan (sec)',fontsize=16)

axs[1].plot([0, 0], [800, 800], color='black', marker='x', label='speedup', linestyle = 'None', markersize=20)
axs[1].legend(bbox_to_anchor=(1., 1), ncol=1,fontsize=16)
fig.suptitle('(number of clients, number of helpers)', y=-0.02,fontsize=18)
    
    
for ax in axs:
    ax.set_xticks(x + width, labels)
    ax.set_ylim(35, 68)
    #ax.set_yticks([i for i in range(35,10,70)],[i for i in range(35,10,70)])
    ax.yaxis.set_ticks([35,45,55,65])
    # set what will actually be displayed at each tick.
    ax.yaxis.set_ticklabels(['$35$','$45$','$55$','$65$'])
    ax.grid(axis = "y") #for grid
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

plt.savefig("granulariy.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()

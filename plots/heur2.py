import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 15})



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

fig, axs = plt.subplots(2,2)
fig.set_size_inches(10, 6)
labels = ('(20,5)', '(20,10)', '(30,5)', '(30,10)', '(50,5)', '(50,10)', '(70,5)', '(70,10)')

props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':30,'shrinkB':30,'linewidth':1}

resnet_1 = {
    'ADMM': (43.97, 40.85, 48.64, 42.41, 56.75, 45.53, 91.06, 53.63),
    'balanced-greedy': (47.0,44.9,51.76,46.15,58.31, 48.96, 91.68, 52.70),
    'random': (52.7,47.0,54.2,52.3, 61.7,55.8,113.6,61.4),
}

resnet_2 = {
    'ADMM': (86.61,60.93,148.8,108.8, 275.07,224.14,401.29,349.06),
    'balanced-greedy': (119.2,89.22,182.3,145.3, 325.9,310.7,452.2,436.9),
    'random': ( 141.4,85.74,186.7,140.5, 333.39,312.06,459.61,436.98),
}

vgg_1 = {
    'ADMM': (24.66,17.6,33.8,28.1, 52.1,36.4,72.9,44.3),
    'balanced-greedy': (54.5,36.4,70.1,41.3,33.233,25.5,37.608,31.83),
    'random': ( 36.1,30.2,39.6,48.0, 66.2, 41.3, 82.6, 49.7),
}

vgg_2 = {
    'ADMM': (98.5,56.4,175.2,108.5,367.0,243.8,558.7,417.0),
    'balanced-greedy': (167.1,118.5,263.0,214.4,454.8,406.2,646.5,597.9),
    'random': (167.179,118.56,216.6,149.6, 483.9,398.0,675.7,608.4),
}

def label_diff(i,j,text,X,Ya, Yb, ii,ij,f,k=0):
    x = (X[i]+2*k+X[i]+j)/2
    y = 1*max(Ya[i], Yb[i])
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':30,'shrinkB':30,'linewidth':1}
    axs[ii][ij].annotate(text, xy=(X[i]+k,y), zorder=11)
    #axs[ii][ij].annotate('', xy=(X[i]+k,y-f), xytext=(X[i]+k+j,y-f), arrowprops=props)



x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars
hatches = ['///', '\\']

multiplier = 0
iter = 0
for attribute, measurement in resnet_1.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[0,0].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        if multiplier == 1:
            rects = axs[0,0].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
        else:
            rects = axs[0,0].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor='firebrick')
    multiplier += 1
    iter += 1

factor = ['16.5\%','13.2\%','7.8\%','19\%',     '8\%','18.4\%','19\%','14.2\%']
admm = resnet_1["ADMM"]
random = resnet_1["random"]
load = resnet_1["balanced-greedy"]
for j in range(0, len(x)):
    axs[0,0].plot([j, j], [admm[j], random[j]], color='black', linestyle = 'dashed', linewidth=1)
    axs[0][0].annotate('', xy=(j,3+random[j]), xytext=(j+0.6,3+random[j]), arrowprops=props)
    #axs[0,0].plot([j, j+0.6], [5+random[j], 5+random[j]], color='black', linewidth=1.5, marker='|')
    axs[0][0].annotate(factor[j], xy=(j,10+random[j]), zorder=20, fontsize=11)

multiplier = 0
iter = 0
for attribute, measurement in vgg_1.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[0,1].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        if multiplier == 1:
            rects = axs[0,1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
        else:
            rects = axs[0,1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor='firebrick')
    multiplier += 1
    iter += 1

factor = ['31.9\%','41.5\%','14.5\%','41.4\%',      '21.3\%','11.7\%','15\%','16.8\%']
admm = vgg_1["ADMM"]
random = vgg_1["random"]
load = vgg_1["balanced-greedy"]
for j in range(0, len(x)):
    max_v = max([random[j], admm[j], load[j]])
    if j < len(x)-4:
        axs[0,1].plot([j, j], [admm[j], max_v], color='black', linestyle = 'dashed', linewidth=1)
        #axs[0,1].plot([j, j+0.6], [5+max_v, 5+max_v], color='black', linewidth=1.5, marker='|')
        axs[0][1].annotate('', xy=(j,3+max_v), xytext=(j+0.6,3+max_v), arrowprops=props)
        axs[0][1].annotate(factor[j], xy=(j,7+max_v), zorder=11, fontsize=11)
    else:
        axs[0,1].plot([j+0.3, j+0.3], [load[j], max_v], color='black', linestyle = 'dashed', linewidth=1)
        #axs[0,1].plot([j+0.3, j+0.6], [5+max_v, 5+max_v], color='black', linewidth=1.5, marker='|')
        axs[0][1].annotate('', xy=(j+0.3,3+max_v), xytext=(j+0.6,3+max_v), arrowprops=props)
        axs[0][1].annotate(factor[j], xy=(j+0.3,7+max_v), zorder=11, fontsize=11)
multiplier = 0
iter = 0
for attribute, measurement in resnet_2.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[1,0].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        if multiplier == 1:
            rects = axs[1,0].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
        else:
            rects = axs[1,0].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor='firebrick')
    multiplier += 1
    iter += 1

factor = ['38.7\%', '28.8\%', '20.2\%', '22.6\%',       '17.4\%', '28.1\%', '12.6\%', '20.1\%']
admm = resnet_2["ADMM"]
random = resnet_2["random"]
load = resnet_2["balanced-greedy"]
for j in range(0, len(x)):
    max_v = max([random[j], admm[j], load[j]])
    axs[1,0].plot([j, j], [admm[j], max_v], color='black', linestyle = 'dashed', linewidth=1)
    #axs[1,0].plot([j, j+0.6], [5+max_v, 5+max_v], color='black', linewidth=1.5, marker='|')
    axs[1][0].annotate('', xy=(j,3+max_v), xytext=(j+0.6,3+max_v), arrowprops=props)
    axs[1][0].annotate(factor[j], xy=(j,30+max_v), zorder=11, fontsize=11)
   
multiplier = 0
iter = 0
for attribute, measurement in vgg_2.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[1,1].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        if multiplier == 1:
            rects = axs[1,1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
        else:
            rects = axs[1,1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor='firebrick')
    multiplier += 1
    iter += 1


factor = ['41.0\%', '52.3\%', '19\%', '27.4\%',       '24.1\%', '38.7\%', '17.3\%', '31.4\%']
admm = vgg_2["ADMM"]
random = vgg_2["random"]
load = vgg_2["balanced-greedy"]
for j in range(0, len(x)):
    max_v = max([random[j], admm[j], load[j]])
    axs[1,1].plot([j, j], [admm[j], max_v], color='black', linestyle = 'dashed', linewidth=1)
    #axs[1,1].plot([j, j+0.6], [10+max_v, 10+max_v], color='black', linewidth=1.5, marker='|')
    axs[1][1].annotate('', xy=(j,3+max_v), xytext=(j+0.6,3+max_v), arrowprops=props)
    axs[1][1].annotate(factor[j], xy=(j,30+max_v), zorder=11, fontsize=11)    

axs[0,1].plot([0, 0], [800, 800], color='black', linewidth=1.5, label='relative gain (\%)')

axs[0,1].legend(bbox_to_anchor=(1, 1.3), ncol=4,fontsize=14)


# Add some text for labels, title and custom x-axis tick labels, etc.
#for ax in axs.flat:

axs[1,0].set_ylabel('batch makespan (sec)', y=1.3,fontsize=20)
fig.suptitle('(number of clients, number of helpers)', y=0.05,fontsize=20)
    
    
maxs = [135, 100, 550,750]    
#resn1, vgg1, resn2, vgg2
ii = 0
for ax in axs[:]:
    for i in range(len(ax)):
        ax[i].set_xticks(x + width, labels)
        ax[i].set_ylim(0, maxs[ii])
        ax[i].grid(axis = "y") #for grid
        ii += 1
        ax[i].tick_params(axis='x', labelsize=10.5)
        ax[i].tick_params(axis='y', labelsize=14)

plt.savefig("heur.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()

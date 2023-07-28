import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 15})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

fig, axs = plt.subplots(1,2)
fig.set_size_inches(10, 3)
labels = ('fwd-prop', 'back-prop')
ind  = np.arange(2)

resnet_par1 = {
    'jetson-gpu': (14.9 , 32),
    'RPI-4': (574.3, 7.6),
    'RPI-3': (642.4, 21),
    'jetson-cpu': (9223.5, 73.4)
}

vgg_par1 = {
    'jetson-gpu': (177.2 , 270.64),
    'RPI-4': (3056.15, 10662.4),
    'RPI-3': (6230, 10862.46),
    'jetson-cpu': (8303.0, 10711.4)
}

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0
iter = 0
hatches = ['\\\\', '..', '++', '+']
for attribute, measurement in resnet_par1.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[0].bar(x + offset, measurement, width, label=attribute, fill=True, color='darkseagreen')
    else:
        rects = axs[0].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor='darkseagreen')
    #axs[0].bar_label(rects, padding=3)
    multiplier += 1
    iter += 1

multiplier = 0
iter = 0
for attribute, measurement in vgg_par1.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[1].bar(x + offset, measurement, width, label=attribute, fill=True, color='darkseagreen')
    else:
        rects = axs[1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor='darkseagreen')
    #axs[1].bar_label(rects, padding=3)
    multiplier += 1
    iter += 1
    
axs[1].legend(bbox_to_anchor=(0.8, 1.2), ncol=4,fontsize=14)


# Add some text for labels, title and custom x-axis tick labels, etc.
#for ax in axs.flat:
axs[0].set_ylabel('Computing time in logscale',fontsize=16)
fig.suptitle('Operation', y=0.02,fontsize=20)
    
    
    

for ax in axs:
    ax.set_xticks(x + width, labels)
    ax.set_yscale('log')
    ax.grid(axis = "y")
    ax.yaxis.set_ticks([1,10,100,1000])
    ax.yaxis.set_ticklabels(['$1$','$10$', '$10^2$','$10^3$'])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

plt.savefig("model_parts.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()

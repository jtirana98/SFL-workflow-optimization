import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 15})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

fig, axs = plt.subplots(1,2)
fig.set_size_inches(6, 3)
labels = ('(50,5)', '(50,10)')
ind  = np.arange(2)

resnet_gran = {
    'It = 200': (64.2, 50),
    'It = 150': (60.9, 49),
    'It = 50': ( 53.7, 49.25),
}

vgg_gran = {
    'It = 200': (54,  46.2),
    'It = 150': (52.5, 45 ),
    'It = 50': (49.95, 40.35),
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
    axs[0].bar_label(rects, padding=3)
    multiplier += 1
    iter += 1

multiplier = 0
iter = 0
for attribute, measurement in vgg_gran.items():
    offset = width * multiplier
    if multiplier == 0:
        rects = axs[1].bar(x + offset, measurement, width, label=attribute, fill=True, color=(0.2, 0.4, 0.6, 0.6))
    else:
        rects = axs[1].bar(x + offset, measurement, width, label=attribute, fill=False, hatch=hatches[iter-1], edgecolor=(0.2, 0.4, 0.6, 0.6))
    axs[1].bar_label(rects, padding=3)
    multiplier += 1
    iter += 1
    
axs[1].legend(bbox_to_anchor=(0.5, 1.2), ncol=3)


# Add some text for labels, title and custom x-axis tick labels, etc.
#for ax in axs.flat:
axs[0].set_ylabel('batch makespan (sec)')
fig.suptitle('(number of clients, number of helpers)', y=0.02)
    
    
    

for ax in axs:
    ax.set_xticks(x + width, labels)
    ax.set_ylim(35, 68)

plt.savefig("granulariy.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


#w_approach = [305, 305, 299, 293, 287, 281, 275, 269, 263, 257, 251, 245, 239, 233, 227, 221, 215, 215, 209, 209, 203, 203, 197, 197, 191, 191, 186, 186, 185, 179]
#maxC = [305, 299, 293, 287, 281, 275, 269, 263, 257, 251, 245, 239, 233, 227, 221, 215, 215, 209, 209, 203, 203, 197, 197, 191, 191, 186, 186, 185, 179, 179]
#w_approach = [305, 305, 299, 293, 287, 281, 275, 269, 263, 257, 251, 245, 239, 233, 227]
#w_approach1 = [305, 305, 304, 303, 300, 296, 281, 275, 264, 262, 257, 257, 257, 250, 250]

#w_approach = [816, 816, 814, 811, 803, 791, 789, 774, 761, 752, 741, 726, 717, 708, 699, 690, 681, 672, 663, 654, 653, 653, 653, 653, 653, 652, 652, 652, 651, 651]
w_approach1 = [816, 816, 814, 811, 803, 791, 789, 774, 761, 752, 741, 726, 717, 708, 699]
w_approach = [816, 814, 813, 812, 796, 792, 774, 769, 757, 744, 734, 724, 715, 706, 697]
#maxC = [816, 814, 811, 803, 791, 789, 774, 761, 752, 741, 726, 717, 708, 699, 690, 681, 672, 663, 654, 653, 653, 653, 653, 653, 652, 652, 652, 651, 651, 651]

#w_approach = [1616, 1614, 1613, 1612, 1604, 1593, 1583, 1573, 1564, 1554, 1543, 1536, 1526, 1515, 1505, 1498, 1487, 1477, 1467, 1458]
#maxC = [1614, 1613, 1612, 1604, 1593, 1583, 1573, 1564, 1554, 1543, 1536, 1526, 1515, 1505, 1498, 1487, 1477, 1467, 1458, 1449]


maxC = []

constraints_1 = [0 for i in range(len(w_approach))]
constraints_2 = [0 for i in range(len(w_approach))]
w_star = 151

fig, (ax1, ax2) = plt.subplots(2)
x_ticks = [i+1 for i in range(len(w_approach))]
print(x_ticks)
ax1.plot(x_ticks, [w_star for i in range(len(x_ticks))], linewidth = 2, marker='o', markersize=2, color="green", label = "Optimal value")
ax1.plot(x_ticks, w_approach1, linestyle='dashed', linewidth = 2, marker='o', markersize=9, color="red", label = "W-approx. exp. 2")
ax1.plot(x_ticks, w_approach, linestyle='dashed', linewidth = 2, marker='o', markersize=5, color="orange", label = "W-approx. exp. 1")

if len(maxC) != 0:
    ax1.plot(x_ticks, maxC, linestyle='dashed', color="black", linewidth = 2, marker='o', markersize=5, label = "max - C")
    #ax2.plot(x_ticks, violation, linestyle = 'None', color="red", marker='+', markersize=12, label = "Violation (T/F)")


ax2.plot(x_ticks, constraints_1, linestyle = 'dashed', linewidth = 2, color="brown", marker='*', markersize=7, label = "constraint-1-violation (%)")
if len(constraints_2) != 0:
    ax2.plot(x_ticks, constraints_2, linestyle = 'None', color="magenta", marker='o', markersize=5, label = "constraint-2-violation (%)")

ax1.set_ylabel("w value")
ax2.set_ylabel("Violations(%)")
plt.xlim(0.9,len(w_approach))

ax1.legend()
ax2.legend()

#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#        fancybox=True, shadow=True, ncol=5)

#ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#        ncol=3, fancybox=True, shadow=True)
plt.show()
'''
Plot examples of excluded shape classes.
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm




#(0) Load results:
dirREPO   = lm.get_repository_path()
dir0      = os.path.join(dirREPO, 'Data', '_ExampleExclusion')
fname00   = os.path.join(dir0, 'cup0.csv')
fname01   = os.path.join(dir0, 'cup1.csv')
fname10   = os.path.join(dir0, 'octopus0.csv')
fname11   = os.path.join(dir0, 'octopus1.csv')
# load:
r00       = np.loadtxt(fname00, delimiter=',', skiprows=1)
r01       = np.loadtxt(fname01, delimiter=',', skiprows=1)
r10       = np.loadtxt(fname10, delimiter=',', skiprows=1)
r11       = np.loadtxt(fname11, delimiter=',', skiprows=1)




#(1) Plot:
plt.close('all')
plt.figure(figsize=(12,4))
axw   = 0.47
ax0   = plt.axes([0,0,axw,1])
ax1   = plt.axes([1-axw,0,axw,1])
c0,c1 = lm.colors[[1,2]]
lw    = 2
ax0.plot(r00[:,0], r00[:,1],     color=c0, lw=lw)
ax0.plot(r01[:,0]+1.1, r01[:,1], color=c1, lw=lw)
ax1.plot(r10[:,0], r10[:,1],     color=c0, lw=lw)
ax1.plot(r11[:,0]+1.1, r11[:,1], color=c1, lw=lw)
for ax in [ax0,ax1]:
	ax.axis('equal')
	ax.axis('off')
labels = '(a)  Cup',  '(b)  Octopus'
[ax.text(0.1, 0.9, label, name='Arial', transform=ax.transAxes, size=16)   for ax,label in zip([ax0,ax1],labels)]
plt.show()





#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'dataset_exclusion.pdf')
plt.savefig(fnamePDF)

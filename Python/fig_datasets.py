
import os
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
from matplotlib import patheffects
import pandas as pd
import lmfree2d as lm



#(0) Load data:
dirREPO    = lm.get_repository_path()
names      = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
R,LM       = [],[]
for name in names:
	fnameXY    = os.path.join(dirREPO, 'Data', name, 'contours.csv')
	fnameLM    = os.path.join(dirREPO, 'Data', name, 'landmarks.csv')
	r          = lm.read_csv(fnameXY)
	frame      = pd.read_csv(fnameLM, sep=',')
	R.append(r)
	LM.append(frame)
templates  = [0, 2, 0,    0, 0, 8,   1, 0, 0]





#(1) Plot:
plt.close('all')
plt.figure(figsize=(14,10))
axx = np.linspace(0, 1, 4)[:3]
axy = np.linspace(0.95, 0, 4)[1:]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])
ax0,ax1,ax2, ax3,ax4,ax5,  ax6,ax7,ax8 = AX.flatten()


lm.set_matplotlib_rcparams()
c0,c1  = lm.colors[2], lm.colors[3]
cmedge = lm.colors[4]
cmface = 'k'

for ax,r,lm,template in zip(AX.flatten(), R, LM, templates):
	for ii,rr in enumerate(r):
		if ii==template:
			ax.fill(rr[:,0], rr[:,1], edgecolor=c0, lw=5, fill=True, facecolor=c1, alpha=0.5, zorder=50)
		else:
			ax.fill(rr[:,0], rr[:,1], edgecolor=c1, lw=1, fill=False, zorder=0)
		if ii==template:
			x,y = lm[ lm['Shape']==(ii+1) ].values[:,2:].T
			ax.plot(x, y, 'o', ms=14, markeredgecolor=cmedge, markerfacecolor=cmface, zorder=51)
			for iii,(xx,yy) in enumerate(zip(x,y)):
				ax.text(xx, yy-0.005, str(iii+1), color='w', zorder=52, size=10, ha='center', va='center')
[ax.axis('equal') for ax in AX.flatten()]

[ax.axis('off')  for ax in AX.flatten()]



### panel labels:
for i,ax in enumerate(AX.flatten()):
	tx = ax.text(0.5, 0.5, '  %s  '%names[i], ha='center', va='center', name='Arial', size=18, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.9), zorder=52)
	tx.set_path_effects([patheffects.withStroke(linewidth=1, foreground=cmedge)])


plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'datasets.pdf')
plt.savefig(fnamePDF)

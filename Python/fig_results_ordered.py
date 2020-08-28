
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects
import lmfree2d as lm



#(0) Load data:
dirREPO    = lm.get_repository_path()
names      = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
R          = [lm.read_csv(os.path.join(dirREPO, 'Data', name, 'contours_sro.csv'))   for name in names]
templates  = [0, 2, 0,    0, 0, 8,   1, 0, 0]



#(1) Plot:
plt.close('all')
lm.set_matplotlib_rcparams()
plt.figure(figsize=(14,10))
axx = np.linspace(0, 1, 4)[:3]
axy = np.linspace(0.95, 0, 4)[1:]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])
ax0,ax1,ax2, ax3,ax4,ax5,  ax6,ax7,ax8 = AX.flatten()


c0,c1  = lm.colors[2], lm.colors[3]
cmedge = lm.colors[4]


for ax,r,template in zip(AX.flatten(), R, templates):
	for ii,rr in enumerate(r):
		if ii==template:
			ax.fill(rr[:,0], rr[:,1], edgecolor=c0, lw=5, fill=True, facecolor=c1, alpha=0.5, zorder=50)
		else:
			ax.fill(rr[:,0], rr[:,1], edgecolor=c1, lw=1, fill=False, zorder=0)
[ax.axis('equal') for ax in AX.flatten()]
[ax.axis('off')  for ax in AX.flatten()]



### panel labels:
for i,ax in enumerate(AX.flatten()):
	tx = ax.text(0.5, 0.5, '  %s  '%names[i], ha='center', va='center', name='Arial', size=18, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.9), zorder=52)
	tx.set_path_effects([patheffects.withStroke(linewidth=1, foreground=cmedge)])


plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'results_ordered.pdf')
plt.savefig(fnamePDF)

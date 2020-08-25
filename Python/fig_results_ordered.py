
import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects
import pandas as pd


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family']      = 'Arial'
plt.rcParams['xtick.labelsize']  = 8
plt.rcParams['ytick.labelsize']  = 8

colors = np.array([
	[177,139,187],
	[166,154,196],
	[132,118,181],
	[225,215,231],
	[252,227,205],
	[231,179,159],
	[213,160,104],
	[166,198,226],
	[134,167,202],
   ]) / 255


def load_geom_and_stack(fnameCSV):
	a         = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	return np.array( [xy[shape==u]  for u in np.unique(shape)] )



#(0) Load data:
dirREPO    = unipath.Path( os.path.dirname(__file__) ).parent
names      = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
R          = [load_geom_and_stack(os.path.join(dirREPO, 'Data', name, 'contours_sro.csv'))   for name in names]
templates  = [0, 2, 0,    0, 0, 8,   1, 0, 0]



#(1) Plot:
plt.close('all')
plt.figure(figsize=(14,10))
plt.get_current_fig_manager().window.move(0, 0)
axx = np.linspace(0, 1, 4)[:3]
axy = np.linspace(0.95, 0, 4)[1:]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])
ax0,ax1,ax2, ax3,ax4,ax5,  ax6,ax7,ax8 = AX.flatten()


c0,c1  = colors[2], colors[3]
cmedge = colors[4]


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

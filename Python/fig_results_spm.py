
'''
Plot results for mass-multivariate (SPM) analysis of the the landmark data.
'''



import os
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import lmfree2d as lm




def custom_legend(ax, colors=None, labels=None, linestyles=None, linewidths=None, markerfacecolors=None, **kwdargs):
	n      = len(colors)
	if linestyles is None:
		linestyles = ['-']*n
	if linewidths is None:
		linewidths = [1]*n
	if markerfacecolors is None:
		markerfacecolors = colors
	x0,x1  = ax.get_xlim()
	y0,y1  = ax.get_ylim()
	h      = [ax.plot([x1+1,x1+2,x1+3], [y1+1,y1+2,y1+3], ls, color=color, linewidth=lw, markerfacecolor=mfc)[0]   for color,ls,lw,mfc in zip(colors,linestyles,linewidths,markerfacecolors)]
	ax.set_xlim(x0, x1)
	ax.set_ylim(y0, y1)
	return ax.legend(h, labels, **kwdargs)
	



#(0) Load results:
dirREPO      = lm.get_repository_path()
names        = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
spm_results  = [lm.read_csv_spm(  os.path.join(dirREPO, 'Data', name, 'spm.csv')  )   for name in names]
snpm_results = [lm.read_csv_spm(  os.path.join(dirREPO, 'Data', name, 'snpm.csv')  )   for name in names]
contours     = [lm.read_csv(  os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')  )   for name in names]



#(1) Plot:
plt.close('all')
lm.set_matplotlib_rcparams()
plt.figure(figsize=(14,10))
# create axes:
axx = np.linspace(0, 1, 4)[:3]
axy = np.linspace(0.95, 0, 4)[1:]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])
# specify constants:
fc0,fc1   = '0.7', '0.9'
ec        = 'k'
vmin,vmax = 30, 150
xoffset   = np.array([1, 0.6, 1,   0.7, 0, 0.6,   1.1, 1, 0])
yoffset   = np.array([0, 0, 0,   0, 0.5, 0,    0.0, 0, 0.5]) * -1
pxoffset  = np.array([0, 0.03, 0,   0, 0.04, 0,   0, 0, -0.25])
pyoffset  = np.array([0, 0, 0.1,   0, 0, 0.25,   -0.17, 0.37, 0])
# plot:
for ax,cont,spm,snpm,xo,yo,pxo,pyo in zip(AX.flatten(), contours, spm_results, snpm_results, xoffset, yoffset, pxoffset, pyoffset):
	cA,cB  = cont[:5], cont[5:]
	spm.plot_with_all_contours(cA, cB, ax, fc=fc0, poffset=(pxo,pyo), vmin=vmin, vmax=vmax)
	snpm.plot_with_all_contours(cA, cB, ax, fc=fc1, offset=(xo,yo), poffset=(pxo,pyo), vmin=vmin, vmax=vmax)
	# spm.plot(ax, fc=fc0, vmin=vmin, vmax=vmax)
	# snpm.plot(ax, fc=fc1, offset=(xo,yo), poffset=(pxo,pyo), vmin=vmin, vmax=vmax)
	ax.axis('off')
# panel labels
[ax.text(0.08, 0.88, '(%s)'%chr(97+i), transform=ax.transAxes, size=16)  for i,ax in enumerate(AX.flatten())]
# colorbar
cbh = plt.colorbar(  AX[0,0].collections[0], cax=plt.axes([0.32, 0.67, 0.015, 0.23])  )
cbh.set_label(r'$T^2$ value', size=16)
# legend:
leg = custom_legend(AX[0,0], colors=['k',lm.colors[5],'k', 'k'], labels=['Mean A','Mean B','Individual Contours',r'Significant Contour Points ( at $\alpha$ = 0.05 )'], linestyles=['-','-','-','o'], linewidths=[3,3,0.5,1], markerfacecolors=[None,None,None,'r'], loc='lower left', bbox_to_anchor=(0.7,0.98), ncol=4)
plt.setp(leg.get_texts(), size=12)


plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'results_spm.pdf')
plt.savefig(fnamePDF)

'''
Plot results for mass-multivariate (SPM) analysis of the the landmark data.
'''



import os,unipath
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family']      = 'Arial'
plt.rcParams['xtick.labelsize']  = 8
plt.rcParams['ytick.labelsize']  = 8



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
	
	
def load_spm_csv(fnameCSV):
	with open(fnameCSV, 'r') as f:
		lines = f.readlines()
	zc    = float( lines[1].strip().split(' = ')[1] ) 
	p     = float( lines[2].strip().split(' = ')[1] ) 
	A     = np.array([s.strip().split(',')   for s in lines[4:]], dtype=float)
	return dict(r0=A[:,:2], r1=A[:,2:4], z=A[:,4], zc=zc, p=p)


def p2str(p):
	return r'$p < 0.001$' if (p < 0.001) else (r'$p = %.3f$' %p)



#(0) Load results:
dirREPO      = unipath.Path( os.path.dirname(__file__) ).parent
names        = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
spm_results  = [load_spm_csv(  os.path.join(dirREPO, 'Data', name, 'spm.csv')  )   for name in names]
snpm_results = [load_spm_csv(  os.path.join(dirREPO, 'Data', name, 'snpm.csv')  )   for name in names]





#(1) Plot:
plt.close('all')
plt.figure(figsize=(14,10))
axx = np.linspace(0, 1, 4)[:3]
axy = np.linspace(0.95, 0, 4)[1:]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])



fc0,fc1   = '0.65', '0.85'
ec        = 'k'
vmin,vmax = 30, 150
xoffset   = np.array([1, 0.6, 1,   0.7, 0, 0.6,   1.1, 1, 0])
yoffset   = np.array([0, 0, 0,   0, 0.5, 0,    0.0, 0, 0.5]) * -1
pxoffset  = np.array([0, 0.03, 0,   0, 0.04, 0,   0, 0, -0.25])
pyoffset  = np.array([0, 0, 0.1,   0, 0, 0.25,   -0.1, 0.35, 0])
for ax,spm,snpm,xo,yo,pxo,pyo in zip(AX.flatten(), spm_results, snpm_results, xoffset, yoffset, pxoffset, pyoffset):
	x0,y0   = spm['r0'].T
	x1,y1   = spm['r1'].T
	ax.fill(x0, y0, color=fc0, zorder=0)
	ax.fill(x0+xo, y0+yo, color=fc1, zorder=0)
	ax.fill(x1, y1, edgecolor=ec, fill=False, zorder=1)
	ax.fill(x1+xo, y1+yo, edgecolor=ec, fill=False, zorder=1)
	
	
	### parametric:
	z,zc,p  = spm['z'], spm['zc'], spm['p']
	if np.any( z > zc ):
		zi  = z.copy()
		zi[ zi < zc] = np.nan
		ax.scatter(x1, y1, s=30, c=zi, cmap='hot', edgecolor='k', vmin=vmin, vmax=vmax, zorder=2)
	ax.text(x0.mean()+pxo, y0.mean()+pyo, p2str(p), ha='center', size=12)
	### nonparametric:
	z,zc,p = snpm['z'], snpm['zc'], snpm['p']
	if np.any( z > zc ):
		zi  = z.copy()
		zi[ zi < zc] = np.nan
		sc = ax.scatter(x1+xo, y1+yo, s=30, c=zi, cmap='hot', edgecolor='k', vmin=vmin, vmax=vmax, zorder=2)
	ax.text(x0.mean()+xo+pxo, y0.mean()+yo+pyo, p2str(p), ha='center', size=12)
	ax.axis('equal')
	ax.axis('off')


# # "parametric" and "nonparametric" labels
# ax = AX[1,1]
# tx0 = ax.text(0.55, 0.32, 'Parametric')
# tx1 = ax.text(0.55, -0.2, 'Nonparametric')
# plt.setp([tx0,tx1], ha='center', size=14, bbox=dict(facecolor='w'))


# panel labels
[ax.text(0.08, 0.88, '(%s)'%chr(97+i), transform=ax.transAxes, size=16)  for i,ax in enumerate(AX.flatten())]


# cbh = plt.colorbar(  sc, cax=plt.axes([0.61, 0.33, 0.02, 0.30])  )
cbh = plt.colorbar(  sc, cax=plt.axes([0.32, 0.67, 0.015, 0.23])  )
cbh.set_label(r'$T^2$ value', size=16)



leg = custom_legend(AX[0,0], colors=[fc0,fc1,ec], labels=['Mean A + parametric results','Mean A + nonparametric results','Mean B'], linestyles=['-']*3, linewidths=[8,8,1], markerfacecolors=None, loc='lower left', bbox_to_anchor=(0.2,0.9))
plt.setp(leg.get_texts(), size=12)


plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'results_spm.pdf')
plt.savefig(fnamePDF)

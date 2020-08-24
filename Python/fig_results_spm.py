
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


def load_spm_csv(fnameCSV):
	with open(fnameCSV, 'r') as f:
		lines = f.readlines()
	zc    = float( lines[1].strip().split(' = ')[1] ) 
	p     = float( lines[2].strip().split(' = ')[1] ) 
	A     = np.array([s.strip().split(',')   for s in lines[4:]], dtype=float)
	r,z   = A[:,:2], A[:,2]
	return dict(r=r, z=z, zc=zc, p=p)
	# def write_csv(fname, r, z, zc, p):
	# 	A  = np.vstack([r.T, z]).T
	# 	with open(fname, 'w') as f:
	# 		f.write('SPM results\n')
	# 		f.write('T2_critical = %.3f\n' %zc)
	# 		f.write('p = %.3f\n' %p)
	# 		# f.write('#Begin geometry\n')
	# 		f.write('X,Y,T2\n')
	# 		for (x,y),zz in zip(r, z):
	# 			f.write('%.6f,%.6f,%.3f\n' %(x,y,zz))


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
plt.get_current_fig_manager().window.move(0, 0)
axx = np.linspace(0, 1, 4)[:3]
axy = np.linspace(0.95, 0, 4)[1:]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])




xoffset  = np.array([1, 0.6, 1,   0.7, 0, 0.6,   1.1, 1, 0])
yoffset  = np.array([0, 0, 0,   0, 0.5, 0,    0.0, 0, 0.5]) * -1
for ax,spm,snpm,xo,yo in zip(AX.flatten(), spm_results, snpm_results, xoffset, yoffset):
	x,y    = spm['r'].T
	ax.fill(x, y, color='0.65', zorder=1)
	ax.fill(x+xo, y+yo, color='0.85', zorder=1)
	### parametric:
	z,zc,p  = spm['z'], spm['zc'], spm['p']
	if np.any( z > zc ):
		zi  = z.copy()
		zi[ zi < zc] = np.nan
		ax.scatter(x, y, s=30, c=zi, cmap='hot', edgecolor='k', vmin=zc, vmax=z.max()+2, zorder=2)
		# ax.text(x.mean(), y.mean(), r'$T^{2*} = %.3f$'%zc, ha='center')
	ax.text(x.mean(), y.mean(), p2str(p), ha='center', size=12)
	### nonparametric:
	z,zc,p = snpm['z'], snpm['zc'], snpm['p']
	if np.any( z > zc ):
		zi  = z.copy()
		zi[ zi < zc] = np.nan
		sc = ax.scatter(x+xo, y+yo, s=30, c=zi, cmap='hot', edgecolor='k', vmin=zc, vmax=z.max()+2, zorder=2)
	ax.text(x.mean()+xo, y.mean()+yo, p2str(p), ha='center', size=12)
	ax.axis('equal')
	ax.axis('off')
	

# # "parametric" and "nonparametric" labels
# ax = AX[0,0]
# tx0 = ax.text(0.47, 0.5, 'Parametric')
# tx1 = ax.text(1.48, 0.5, 'Nonparametric')
# plt.setp([tx0,tx1], ha='center', size=14, bbox=dict(facecolor='w'))

# "parametric" and "nonparametric" labels
ax = AX[1,1]
tx0 = ax.text(0.55, 0.32, 'Parametric')
tx1 = ax.text(0.55, -0.2, 'Nonparametric')
plt.setp([tx0,tx1], ha='center', size=14, bbox=dict(facecolor='w'))


# panel labels
[ax.text(0.08, 0.88, '(%s)'%chr(97+i), transform=ax.transAxes, size=16)  for i,ax in enumerate(AX.flatten())]


cbh = plt.colorbar(  sc, cax=plt.axes([0.61, 0.33, 0.02, 0.30])  )
cbh.set_label(r'$T^2$ value', size=16)

#
# ### panel labels:
# for i,ax in enumerate(AX.flatten()):
# 	tx = ax.text(0.5, 0.5, '  %s  '%names[i], ha='center', va='center', name='Arial', size=18, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.9), zorder=52)
# 	tx.set_path_effects([patheffects.withStroke(linewidth=1, foreground=cmedge)])


plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'results_spm.pdf')
plt.savefig(fnamePDF)

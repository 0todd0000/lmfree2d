
import os
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import lmfree2d as lm

# plt.rcParams["font.family"] = "Times"
# plt.rcParams["mathtext.fontset"] = "stix"



def plot_xy(ax, xy, color):
	h = []
	for a in xy:
		a = np.vstack([a,a[0]])
		hh = ax.plot(a[:,0], a[:,1], color=color, lw=0.5)
		h.append(hh[0])
	return h

def plot_spm(ax, spmi, template_contour, vmin=0, vmax=50):
	m       = template_contour
	mtmp    = np.vstack( [m,m[0]] )
	ax.plot(mtmp[:,0], mtmp[:,1], 'k-', label='Grand mean contour')
	i       = spmi.z > spmi.zstar
	# ax.plot(m[i,0], m[i,1], 'ro')
	if i.sum() > 0:
		ax.scatter(m[i,0], m[i,1], s=30, c=spmi.z[i], cmap='hot', edgecolor='k', alpha=1, vmin=vmin, vmax=vmax, zorder=10, label='Suprathreshold point')
	



#(0) Load data:
dirREPO      = lm.get_repository_path()
fnameNPZ     = os.path.join(dirREPO, 'Data', '_Appendix', 'contours.npz')
with np.load( fnameNPZ ) as Z:
	r0,r,xy0,xy,size = [Z[s]  for s in ['r0','r','xy0','xy', 'size']]
J = r.shape[0]
n = int( J/2 )



#(1) Run ANCOVA:
# specify design matrix:
J          = r.shape[0]   # sample size
n          = int( J/2 )   # half-sample size
X          = np.zeros( (J,3) )
X[:,0]     = size    # size covariate (continuous)
X[:n,1]    = 1       # group A (binary)
X[n:,2]    = 1       # group B (binary)
# specify contrast vectors:
c0         = [1,  0, 0]   # size-related effects
c1         = [0, -1, 1]   # group-related effects
# conduct tests (using polar shape representations for simplicity):
alpha      = 0.05
t00i       = spm1d.stats.glm(r0, X, c0).inference(alpha)
t01i       = spm1d.stats.glm(r0, X, c1).inference(alpha)
t0i        = spm1d.stats.glm(r, X, c0).inference(alpha)
t1i        = spm1d.stats.glm(r, X, c1).inference(alpha)



#(2) Plot:
plt.close('all')
fig,AX = plt.subplots( 3, 2, figsize=(8,9) )
ax0,ax1,ax2,ax3,ax4,ax5 = AX.flatten()
c0,c1  = 'b', 'r'

# plot original contours
h0 = plot_xy(ax0, xy0[:n], c0)[0]
h1 = plot_xy(ax0, xy0[n:], c1)[1]

ax0.axis('equal')
ax0.legend([h0,h1], ['Group A', 'Group B'], loc='upper right')
ax1.legend([h0,h1], ['Group A', 'Group B'], loc='upper right')

# plot registered contours
plot_xy(ax1, xy[:n], c0)
plot_xy(ax1, xy[n:], c1)
ax1.axis('equal')

# plot size-related results:
m       = xy.mean(axis=0)
vmin,vmax = 0, 60
plot_spm(ax2, t00i, m, vmin, vmax)
plot_spm(ax3, t0i, m, vmin, vmax)
ax2.legend(loc='center')
ax3.legend(loc='center')

# plot group-related results:
m       = xy.mean(axis=0)
plot_spm(ax4, t01i, m, vmin, vmax)
plot_spm(ax5, t1i, m, vmin, vmax)
ax4.legend(loc='center')
ax5.legend(loc='center')

# plot colorbar:
cbh = plt.colorbar(  ax2.collections[0], cax=plt.axes([0.46, 0.37, 0.015, 0.23])  )
cbh.set_label(r'$\sqrt{T^2}$ value', size=16)

# set axes properties:
for ax in AX.flatten():
	ax.axis('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xticklabels(ax.get_xticks())
	ax.set_yticklabels(ax.get_yticks())
	for spine in ax.spines.values():
		spine.set_color('1.0')

# annotate signals:
bbox = dict(fc='w', ec='0.9')
ax0.annotate('Group-dependent signal', xy=(0, 8), xytext=(19, -19), ha='center', arrowprops=dict(facecolor='black', shrink=0.05), bbox=bbox)
ax0.annotate('Size-dependent signal', xy=(-21, 0), xytext=(-19, -19), ha='center', arrowprops=dict(facecolor='black', shrink=0.05), bbox=bbox)
ax1.annotate('Group-dependent signal', xy=(0, 11), xytext=(0, 5), ha='center', arrowprops=dict(facecolor='black', shrink=0.05), bbox=bbox)
ax1.annotate('Size-dependent signal', xy=(-11, 0), xytext=(0, -2), ha='center', arrowprops=dict(facecolor='black', shrink=0.05), bbox=bbox)


# panel and column labels:
sz = 16
[ax.text(0.05, 0.9, '(%s)' %chr(97+i), size=sz, transform=ax.transAxes)  for i,ax in enumerate(AX.flatten())]
ax0.set_title('Original', size=sz)
ax1.set_title('Registered', size=sz)
ax0.set_ylabel('Contour dataset', size=sz)
ax2.set_ylabel('Size effect', size=sz)
ax4.set_ylabel('Group effect', size=sz)

plt.tight_layout()
plt.show()



#(3) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'appendix_ancova.pdf')
plt.savefig(fnamePDF)


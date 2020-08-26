
import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import pycpd



def register_cpd_single_pair(r0, r1):
	reg     = pycpd.RigidRegistration(X=r0, Y=r1)	
	reg.register()
	r1r     = reg.TY
	return r1r


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




#(0) Load results:
dirREPO    = unipath.Path( os.path.dirname(__file__) ).parent
names      = ['Bell', 'Comma', 'Cup', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
name       = 'Bell'
fname      = os.path.join(dirREPO, 'Data', name, 'contours.csv')
a          = np.loadtxt(fname, delimiter=',', skiprows=1)
shape      = np.asarray(a[:,0], dtype=int)
xy         = a[:,1:]
r0,r1      = xy[shape==2], xy[shape==1]
r1r        = register_cpd_single_pair(r0, r1)


# s1.set_scale(1.5)



#(1) Plot:
plt.close('all')
fig,AX = plt.subplots( 1, 2, figsize=(8,3) )
plt.get_current_fig_manager().window.move(0, 0)
ax0,ax1 = AX.flatten()

ax0.plot(r0[:,0], r0[:,1], 'b.')
ax0.plot(r1[:,0], r1[:,1], 'r.')
ax0.text(0, 1, '(a)  Original data', size=14, transform=ax0.transAxes)
ax0.text(0.8, 0.76, f'nPoints = {r0.shape[0]}', size=10, color='b', transform=ax0.transAxes)
ax0.text(0.8, 0.65, f'nPoints = {r1.shape[0]}', size=10, color='r', transform=ax0.transAxes)
ax0.axis('equal')
ax0.axis('off')


ax1.plot(r0[:,0], r0[:,1], 'b.')
ax1.plot(r1r[:,0], r1r[:,1], 'r.')
ax1.text(0.1, 1, '(b)  CPD-registered', size=14, transform=ax1.transAxes)
ax1.axis('equal')
ax1.axis('off')

plt.show()




# #(2) Save figure:
# fnamePDF  = os.path.join(dirREPO, 'Figures', 'cpd.pdf')
# plt.savefig(fnamePDF)
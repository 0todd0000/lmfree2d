
import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm




#(0) Load results:
dirREPO    = lm.get_repository_path()
fname0     = os.path.join(dirREPO, 'Data', '_ExampleCPD', 'contour0.csv')
fname1     = os.path.join(dirREPO, 'Data', '_ExampleCPD', 'contour1.csv')
r0         = np.loadtxt(fname0, delimiter=',', skiprows=1)
r1         = np.loadtxt(fname1, delimiter=',', skiprows=1)
r1r        = lm.register_cpd_single_pair(r0, r1)



#(1) Plot:
plt.close('all')
lm.set_matplotlib_rcparams()
plt.figure(figsize=(6,4))
axw   = 0.47
axh   = 0.90
ax0   = plt.axes([0,0,axw,axh])
ax1   = plt.axes([1-axw,0,axw,axh])
c0,c1 = lm.colors[[2,5]]

ax0.plot(r0[:,0], r0[:,1], 'o', ms=4, color=c0)
ax0.plot(r1[:,0], r1[:,1], 'o', ms=4, color=c1)
ax0.text(0.5, 1.05, '(a)  Original', size=14, transform=ax0.transAxes, ha='center')
ax0.text(0.92, 0.76, f'nPoints = {r0.shape[0]}', size=12, color=c0, transform=ax0.transAxes)
ax0.text(0.92, 0.65, f'nPoints = {r1.shape[0]}', size=12, color=c1, transform=ax0.transAxes)
ax0.axis('equal')
ax0.axis('off')

ax1.plot(r0[:,0], r0[:,1], 'o', ms=4, color=c0)
ax1.plot(r1r[:,0], r1r[:,1], 'o', ms=4, color=c1)
ax1.text(0.5, 1.05, '(b)  CPD-registered', size=14, transform=ax1.transAxes, ha='center')
ax1.axis('equal')
ax1.axis('off')

plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'cpd.pdf')
plt.savefig(fnamePDF)
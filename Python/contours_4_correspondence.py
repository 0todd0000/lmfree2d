
'''
Find optimum correspondence between shapes
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm




# #(0) Find optimum-roll correspondence between two sets of contour points:
# dirREPO   = lm.get_repository_path()
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sro.csv')
# r         = lm.read_csv(fname0)
# r0        = r[0]
# r1        = r[1]
# n         = max(r0.shape[0], r1.shape[0])
# r0        = lm.set_npoints(r0, n)
# r1        = lm.set_npoints(r1, n)
# r2        = lm.corresp_roll(r0, r1)
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 3, figsize=(12,3) )
# ax0,ax1,ax2 = AX.flatten()
# ax0.scatter(r0[:,0], r0[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# ax1.scatter(r1[:,0], r1[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# ax2.scatter(r2[:,0], r2[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# [ax.axis('equal') for ax in AX]
# plt.show()


# #(1) Find optimum-order correspondence (one dataset):
# dirREPO     = lm.get_repository_path()
# name        = 'Bell'
# fname0      = os.path.join(dirREPO, 'Data', name, 'contours_sro.csv')
# r0          = lm.read_csv(fname0)
# rtemplate,n = lm.get_shape_with_most_points(r0)
# r1          = [lm.set_npoints(rr, n) for rr in r0]
# r1          = [lm.corresp_roll(rtemplate, rr)  for rr in r1]
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 2, 5, figsize=(12,6) )
# for ax,rr in zip(AX.flatten(), r1):
# 	ax.scatter(rr[:,0], rr[:,1], c=np.arange(rr.shape[0]), cmap='jet', vmin=0, vmax=n)
# 	ax.axis('equal')
# plt.show()



#(2) Process all datasets:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'Finding roll correspondence for {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sro.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')
	r0        = lm.read_csv(fname0)
	rtemp,n   = lm.get_shape_with_most_points(r0)
	r1        = [lm.set_npoints(rr, n) for rr in r0]
	r1        = [lm.corresp_roll(rtemp, rr)  for rr in r1]
	lm.write_csv(fname1, r1)




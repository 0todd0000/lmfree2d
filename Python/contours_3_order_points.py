
'''
Order contour points in a clockwise manner
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm





# #(0) Order single set of contour points:
# dirREPO   = lm.get_repository_path()
# name      = 'Comma'
# fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sr.csv')
# r         = lm.read_csv(fname0)
# r0        = r[0]
# r1        = lm.reorder_points(r0, optimum_order=True, ensure_clockwise=True)
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# ax0.fill(r0[:,0], r0[:,1], fill=False)
# ax1.fill(r1[:,0], r1[:,1], fill=False)
# ax0.axis('equal')
# ax1.axis('equal')
# plt.show()





# #(1) Order points for one dataset:
# dirREPO   = lm.get_repository_path()
# names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
# name      = names[0]
# fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sr.csv')
# r0        = lm.read_csv(fname0)
# r1        = [lm.reorder_points(rr, optimum_order=True, ensure_clockwise=True)  for rr in r0]
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# [ax0.fill(rr[:,0], rr[:,1], fill=False, edgecolor='0.7', lw=0.5)  for rr in r0]
# [ax1.fill(rr[:,0], rr[:,1], fill=False, edgecolor='0.7', lw=0.5)  for rr in r1]
# ax0.axis('equal')
# ax1.axis('equal')
# plt.show()




#(2) Order points for all datasets:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'Ordering points for {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sr.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'contours_sro.csv')
	r0        = lm.read_csv(fname0)
	r1        = [lm.reorder_points(rr, optimum_order=True, ensure_clockwise=True)  for rr in r0]
	lm.write_csv(fname1, r1)



'''
Process a single dataset and plot the results.

Results for different processing stages are plotted in separate figures.
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm




#(0) Process a single dataset:
### load data
dirREPO     = lm.get_repository_path()
names       = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
name        = names[0]
fname0      = os.path.join(dirREPO, 'Data', name, 'contours.csv')
r0          = lm.read_csv(fname0)
np.random.seed( 10 + names.index(name) )
r0          = lm.shuffle_points(r0)
### CPD registration
rtemp0,n,i  = lm.get_shape_with_most_points(r0)
r1          = lm.register_cpd(r0, rtemp0)
### reorder points:
r2          = lm.reorder_points(r1, optimum_order=True, ensure_clockwise=True)
### correspondence:
rtemp1      = r2[i]
r3          = lm.set_npoints(r2, n)
r4          = lm.corresp_roll(r3, rtemp1)
### hypothesis test:
rA,rB       = r4[:5], r4[5:]
results     = lm.two_sample_test(rA, rB, parametric=True)
print(results)




#(1) Plot all results:
plt.close('all')
# plot original, shuffled contour points:
plt.figure(figsize=(12,6))
ax = plt.axes([0,0,1,0.9])
lm.plot_point_order(ax, r0)
ax.set_title('Original (shuffled) contour points', size=20)
plt.show()
# plot registration results
plt.figure(figsize=(8,4))
ax0 = plt.axes([0,0,0.5,0.9])
ax1 = plt.axes([0.5,0,0.5,0.9])
lm.plot_registration(ax0, r0, rtemp0)
lm.plot_registration(ax1, r1, rtemp0)
ax0.set_title('Before Registration', size=14)
ax1.set_title('After Registration', size=14)
# plot point ordering:
plt.figure(figsize=(12,6))
ax = plt.axes([0,0,1,0.9])
lm.plot_point_order(ax, r2)
ax.set_title('Re-ordered points', size=20)
# plot corrspondence:
plt.figure(figsize=(12,6))
ax = plt.axes([0,0,1,0.9])
lm.plot_correspondence(ax, r4, rtemp1)
ax.set_title('Correspondence', size=20)
plt.show()
# plot hypothesis testing results:
plt.figure(figsize=(6,4))
ax = plt.axes([0,0,1,0.9])
results.plot(ax)
ax.legend()
ax.set_title('Hypothesis test results', size=14)
plt.show()





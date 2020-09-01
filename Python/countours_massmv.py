
'''
Shuffle the point order for all shapes
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm

import importlib;  importlib.reload(lm)


#(0) Process a single dataset:
### load data
dirREPO     = lm.get_repository_path()
names       = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
name        = names[8]
fname0      = os.path.join(dirREPO, 'Data', name, 'contours.csv')
r0          = lm.read_csv(fname0)
### CPD registration
rtemplate,n = lm.get_shape_with_most_points(r0)
r1          = lm.register_cpd(r0, rtemplate)
r2          = lm.reorder_points(r1, optimum_order=True, ensure_clockwise=True)
r3          = lm.set_npoints(r2, n)
r4          = lm.corresp_roll(r3, rtemplate)
### hypothesis test:
rA,rB       = r4[:5], r4[5:]
results     = lm.two_sample_test(rA, rB, parametric=True)
print(results)
### plot:
plt.close('all')
plt.figure()
ax        = plt.axes()
results.plot(ax)
ax.legend()
plt.show()



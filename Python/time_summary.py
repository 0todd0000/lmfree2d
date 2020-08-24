
'''
Summarize execution time results.
'''

import os,unipath
import numpy as np
import pandas as pd



# Landmarks, Procrustes ANOVA:
dirREPO  = unipath.Path( os.path.dirname(__file__) ).parent
fnameCSV = os.path.join(dirREPO, 'Results', 'time_landmarks_procANOVA.csv')
df       = pd.read_csv(fnameCSV)
labels   = df.keys()[1:]
m        = [1000*df[s].mean() for s in labels]
sd       = [1000*df[s].std(ddof=1) for s in labels]
print('\n\n\nLandmarks, Scalar')
print(labels)
print('Mean:  %.1f, %.1f' %tuple(m) )
print('SD:    %.1f, %.1f' %tuple(sd) )



# Landmarks, mass multivariate:
dirREPO  = unipath.Path( os.path.dirname(__file__) ).parent
fnameCSV = os.path.join(dirREPO, 'Results', 'time_landmarks_massmv.csv')
df       = pd.read_csv(fnameCSV) 
labels   = df.keys()[1:]
m        = [1000*df[s].mean() for s in labels]
sd       = [1000*df[s].std(ddof=1) for s in labels]
print('\n\n\nLandmarks, MassMV')
print(labels)
print('Mean:  %.1f, %.1f' %tuple(m) )
print('SD:    %.1f, %.1f' %tuple(sd) )



# Contours, Procrustes ANOVA:
dirREPO  = unipath.Path( os.path.dirname(__file__) ).parent
fnameCSV = os.path.join(dirREPO, 'Results', 'time_full_ProcANOVA.csv')
df       = pd.read_csv(fnameCSV)
labels   = df.keys()[1:]
m        = [1000*df[s].mean() for s in labels]
sd       = [1000*df[s].std(ddof=1) for s in labels]
print('\n\n\nContours, Scalar')
print(labels)
print('Mean:  %.1f, %.1f' %tuple(m) )
print('SD:    %.1f, %.1f' %tuple(sd) )



# Contours, mass multivariate:
dirREPO  = unipath.Path( os.path.dirname(__file__) ).parent
fnameCSV = os.path.join(dirREPO, 'Results', 'time_contours_massmv.csv')
df       = pd.read_csv(fnameCSV)
labels   = df.keys()[1:]
m        = [1000*df[s].mean() for s in labels]
sd       = [1000*df[s].std(ddof=1) for s in labels]
print('\n\n\nContours, MassMV')
print(labels)
print('Mean:  %.1f, %.1f' %tuple(m) )
print('SD:    %.1f, %.1f' %tuple(sd) )

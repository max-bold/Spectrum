import numpy as np

a = np.array((0.75,0.125,1.125,0.997))
print(np.rint(a))
print(np.astype(a,int))
print(type(np.rint(a)[0]))
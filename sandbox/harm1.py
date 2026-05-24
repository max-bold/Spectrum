import numpy as np

x= np.linspace(0,2*np.pi,500)
y1 = np.sin(x)
y2 = y1**2
y3 = y1**3

import matplotlib.pyplot as plt
plt.plot(x,y1,label='sin(x)')
plt.plot(x,(y2-0.5)*2,label='sin^2(x)')
plt.plot(x,(y3-y1*3/4)*4,label='sin^3(x)')
plt.legend()
plt.show()
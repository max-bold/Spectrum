import scipy.signal.windows as ssw
import matplotlib.pyplot as plt
w=ssw.blackman(400)
plt.plot(w)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = y = np.arange(-4, 4, 0.1)
x, y = np.meshgrid(x,y)
plt.contour(x, y, x**2/9 + y**2-1, [0])

plt.axis('scaled')
plt.show()




import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple plot")

plt.legend()

plt.show()

M = np.array([[1, 0, 0],
              [0, 0, -1],
              [0, 1, 0]])

print(np.linalg.inv(M))
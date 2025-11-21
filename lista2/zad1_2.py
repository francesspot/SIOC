import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Powierzchnia 3D (Surface plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title("Wykres powierzchni 3D")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Kontury 2D (Contour plot)
plt.contourf(X, Y, Z, cmap='plasma')
plt.title("Wykres konturów")
plt.colorbar()
plt.show()
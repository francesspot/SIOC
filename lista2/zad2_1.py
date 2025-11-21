import numpy as np
import matplotlib.pyplot as plt

n = 64
r = 20
X, Y = np.ogrid[:n, :n]
srodek = n // 2
odleglosc_punktów = np.sqrt((X - srodek)**2 + (Y - srodek)**2)
macierz = (odleglosc_punktów < r).astype(int)

plt.imshow(macierz, cmap="gray")
plt.title("Macierz z okręgiem w środku")
plt.show()
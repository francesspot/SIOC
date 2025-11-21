import numpy as np

x = np.linspace(0, 10, 1000)
f1 = -x * np.exp(-x / 2)
f2 = np.sin(np.pi * x) + 2 * np.cos(2 * np.pi * x) + 3 * np.sin(2 * np.pi * x) * np.exp(-x / 2)
f3 = np.where(x > 2, 2 * x * np.exp(-x), np.nan)

def dlugosc(x, y):
    poprawna_liczba = ~np.isnan(y)
    x = x[poprawna_liczba]
    y = y[poprawna_liczba]
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2))

print("Długość krzywej f1:", round(dlugosc(x, f1), 2))
print("Długość krzywej f2:", round(dlugosc(x, f2), 2))
print("Długość krzywej f3:", round(dlugosc(x, f3), 2))

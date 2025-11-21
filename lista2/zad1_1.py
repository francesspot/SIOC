import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

f1 = -x * np.exp(-x / 2)
f2 = np.sin(np.pi * x) + 2 * np.cos(2 * np.pi * x) + 3 * np.sin(2 * np.pi * x) * np.exp(-x / 2)
f3 = np.where(x > 2, 2 * x * np.exp(-x), np.nan)

# Wszystkie wykresy na jednym obrazie
plt.plot(x, f1, label="f1(x)", color="blue")
plt.plot(x, f2, label="f2(x)", color="orange")
plt.plot(x, f3, label="f3(x)", color="green")
plt.legend()
plt.title("Wykres wszystkich funkcji na jednym obrazie")
plt.show()


# Kazdy wykres osobno (podwykresy)
plt.subplot(3, 1, 1)
plt.plot(x, f1)
plt.title("Wykres funkcji f1(x)")

plt.subplot(3, 1, 2)
plt.plot(x, f2)
plt.title("Wykres funkcji f2(x)")

plt.subplot(3, 1, 3)
plt.plot(x, f3)
plt.title("Wykres funkcji f3(x)")

plt.tight_layout()
plt.show()

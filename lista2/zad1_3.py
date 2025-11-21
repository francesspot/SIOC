import matplotlib.pyplot as plt
from skimage import io

img = io.imread("lista2/pug.jpg")
plt.imshow(img)
plt.title("Pug")
plt.axis("off")
plt.show()
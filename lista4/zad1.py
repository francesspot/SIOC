import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convolve2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    img = np.zeros((H + 2*pad_h, W + 2*pad_w), dtype=np.float32)
    img[pad_h:pad_h+H, pad_w:pad_w+W] = image

    out = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            region = img[y:y+kH, x:x+kW]
            out[y, x] = np.sum(region * kernel)

    return out



Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

W = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]], dtype=np.float32)


image_path = "lista4/pug.jpg"

img = Image.open(image_path).convert("L")
image = np.array(img)
image = image.astype(np.float32) / 255.0

gx = convolve2d(image, Sx)
gy = convolve2d(image, Sy)

edges = np.sqrt(gx*gx + gy*gy)

edges = edges / (edges.max() + 1e-12)
edges_out = (edges * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Sobel X (gx)")
plt.imshow(np.abs(gx), cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Sobel Y (gy)")
plt.imshow(np.abs(gy), cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Krawędzie (magnitude)")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

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

Scharr_x = np.array([[ -3,  0,  3], [-10,  0, 10], [ -3,  0,  3]], dtype=np.float32)

Scharr_y = np.array([[ -3, -10, -3], [  0,   0,  0], [  3,  10,  3]], dtype=np.float32)

Prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)

Prewitt_y = np.array([[1,  1,  1], [0,  0,  0], [-1, -1, -1]], dtype=np.float32)

G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

box_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0

unsharp_masking = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=np.float32) / -256.0

W = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]], dtype=np.float32)


image_path = "lista4/pug.jpg"

img = Image.open(image_path).convert("L")
image = np.array(img)
image = image.astype(np.float32) / 255.0

# Wykrywanie krawędzi za pomocą operatora Sobela
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

#  Wykrywanie krawędzi za pomocą detektora Sobela (suma wartości bezwzględnych)
sobel_sum = np.abs(gx) + np.abs(gy)
sobel_sum = sobel_sum / (sobel_sum.max() + 1e-12)
sobel_sum_out = (sobel_sum * 255).astype(np.uint8)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Detektor Sobela (|gx| + |gy|)")
plt.imshow(sobel_sum, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()


# Wykrywanie krawędzi za pomocą operatora Laplace'a
laplace = convolve2d(image, L)
laplace_abs = np.abs(laplace)
laplace_edges = laplace_abs / (laplace_abs.max() + 1e-12)
laplace_out = (laplace_edges * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Krawędzie (Laplace)")
plt.imshow(laplace_edges, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Wykrywanie krawędzi za pomocą operatora Scharra
gx_scharr = convolve2d(image, Scharr_x)
gy_scharr = convolve2d(image, Scharr_y)

edges_scharr = np.sqrt(gx_scharr*gx_scharr + gy_scharr*gy_scharr)
edges_scharr = edges_scharr / (edges_scharr.max() + 1e-12)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Scharr X (gx)")
plt.imshow(np.abs(gx_scharr), cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Scharr Y (gy)")
plt.imshow(np.abs(gy_scharr), cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Krawędzie (Scharr)")
plt.imshow(edges_scharr, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Wykrywanie krawędzi za pomocą operatora Prewitta
gx_prewitt = convolve2d(image, Prewitt_x)
gy_prewitt = convolve2d(image, Prewitt_y)

edges_prewitt = np.sqrt(gx_prewitt*gx_prewitt + gy_prewitt*gy_prewitt)
edges_prewitt = edges_prewitt / (edges_prewitt.max() + 1e-12)
edges_prewitt_out = (edges_prewitt * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Prewitt X (gx)")
plt.imshow(np.abs(gx_prewitt), cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Prewitt Y (gy)")
plt.imshow(np.abs(gy_prewitt), cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Krawędzie (Prewitt)")
plt.imshow(edges_prewitt, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Porównanie metod wykrywania krawędzi: Oryginał vs Sobel vs Detektor Sobela vs Laplace vs Scharr vs Prewitt
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Krawędzie Sobel")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Detektor Sobela (|gx| + |gy|)")
plt.imshow(sobel_sum, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Krawędzie (Laplace)")
plt.imshow(laplace_edges, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Krawędzie (Scharr)")
plt.imshow(edges_scharr, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title(" Krawędzie (Prewitt)")
plt.imshow(edges_prewitt, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Rozmycie Gaussa
blurred = convolve2d(image, G)
blurred = np.clip(blurred, 0, 1)
blurred = (blurred * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rozmycie Gaussa")
plt.imshow(blurred, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Rozmycie pudełkowe
box_blurred = convolve2d(image, box_blur)
box_blurred = np.clip(box_blurred, 0, 1)
box_blurred = (box_blurred * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rozmycie pudełkowe")
plt.imshow(box_blurred, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Maskowanie nieostrości
unsharp_blurred = convolve2d(image, unsharp_masking)
unsharp_blurred = np.clip(unsharp_blurred, 0, 1)
unsharp_blurred = (unsharp_blurred * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rozmycie maskowaniem nieostrości")
plt.imshow(unsharp_blurred, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Porównanie oryginału, rozmycia Gaussa, rozmycia pudełkowego oraz maskowania nieostrości
plt.figure(figsize=(18, 10))

plt.subplot(2, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Rozmycie Gaussa")
plt.imshow(blurred, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Rozmycie pudełkowe")
plt.imshow(box_blurred, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Maskowanie nieostrości")
plt.imshow(unsharp_blurred, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Wyostrzenie obrazu
sharpened = convolve2d(image, W)
sharpened = np.clip(sharpened, 0, 1)
sharpened = (sharpened * 255).astype(np.uint8)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Wyostrzenie obrazu")
plt.imshow(sharpened, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# DEMOZAIKOWANIE

# Kernele do interpolacji
kernel_G = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]], dtype=np.float32) / 4.0

kernel_R = np.array([[1, 0, 1],[0, 0, 0],[1, 0, 1]], dtype=np.float32) / 4.0

kernel_B = np.array([[1, 0, 1],[0, 0, 0],[1, 0, 1]], dtype=np.float32) / 4.0

H, W = image.shape

# Inicjalizacja masek dla kanałów R, G, B
mask_R = np.zeros_like(image, dtype=np.float32)
mask_G = np.zeros_like(image, dtype=np.float32)
mask_B = np.zeros_like(image, dtype=np.float32)

# G R
# B G

# Ustawienie masek zgodnie z wzorem Bayera
mask_G[0::2, 0::2] = 1 # G w lewym górnym rogu
mask_G[1::2, 1::2] = 1 # G w prawym dolnym rogu
mask_R[0::2, 1::2] = 1 # R w prawym górnym rogu
mask_B[1::2, 0::2] = 1 # B w lewym dolnym rogu

# Ekstrakcja znanych wartości dla każdego kanału
R = image * mask_R
G = image * mask_G
B = image * mask_B

# Interpolacja brakujących wartości
R_interp = convolve2d(R, kernel_R)
G_interp = convolve2d(G, kernel_G)
B_interp = convolve2d(B, kernel_B)

# Uzupełnienie brakujących wartości oryginalnymi danymi
R_full = np.where(R >= 0, R, R_interp)
G_full = np.where(G >= 0, G, G_interp)
B_full = np.where(B >= 0, B, B_interp)

# Scalanie kanałów w obraz RGB
demosaiced_image = np.stack((R_full, G_full, B_full), axis=-1)
demosaiced_image = np.clip(demosaiced_image, 0, 1)
demosaiced_image = (demosaiced_image * 255).astype(np.uint8)

plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(Image.open(image_path))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Demozaikowanie filtrem Bayera")
plt.imshow(demosaiced_image)
plt.axis("off")
plt.tight_layout()
plt.show()


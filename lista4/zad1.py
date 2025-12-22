import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

def convolve2d(image, kernel, padding_mode="reflect"):
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h_top = (kH - 1) // 2
    pad_h_bottom = kH // 2
    
    pad_w_left = (kW - 1) // 2
    pad_w_right = kW // 2

    img = np.pad(image, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), mode=padding_mode)

    kernel = np.flipud(np.fliplr(kernel))

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

G = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]], dtype=np.float32) / 16.0

box_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0

G_5x5 = np.array([[1, 4,  6,  4,  1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4,  6,  4,  1]], dtype=np.float32) / 256.0

W = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]], dtype=np.float32)

unsharp_masking = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=np.float32) / -256.0


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
plt.title("Krawędzie (magnituda)")
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

plt.figure(figsize=(12, 6))
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

plt.figure(figsize=(12, 6))

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

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rozmycie pudełkowe (Box blur)")
plt.imshow(box_blurred, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Rozmycie Gaussa 5x5
blurred_5x5 = convolve2d(image, G_5x5)
blurred_5x5 = np.clip(blurred_5x5, 0, 1)
blurred_5x5 = (blurred_5x5 * 255).astype(np.uint8)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rozmycie Gaussa 5x5")
plt.imshow(blurred_5x5, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Porównanie oryginału, rozmycia Gaussa, rozmycia pudełkowego i rozmycia Gaussa 5x5
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Rozmycie Gaussa")
plt.imshow(blurred, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Rozmycie pudełkowe (Box blur)")
plt.imshow(box_blurred, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Rozmycie Gaussa 5x5")
plt.imshow(blurred_5x5, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Wyostrzenie obrazu
sharpened = convolve2d(image, W)
sharpened = np.clip(sharpened, 0, 1)
sharpened = (sharpened * 255).astype(np.uint8)

plt.figure(figsize=(12, 6))

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

# Maskowanie nieostrości (Unsharp Masking)
unsharp_masked = convolve2d(image, unsharp_masking)
unsharp_masked = np.clip(unsharp_masked, 0, 1)
unsharp_masked = (unsharp_masked * 255).astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Maskowanie nieostrości (Unsharp Masking)")
plt.imshow(unsharp_masked, cmap='gray')
plt.axis("off")
plt.tight_layout()
plt.show()

# Porównanie oryginału, wyostrzenia, maskowania nieostrości
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.title("Oryginał")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Wyostrzenie obrazu")
plt.imshow(sharpened, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Maskowanie nieostrości (Unsharp Masking)")
plt.imshow(unsharp_masked, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================
# DEMOZAIKOWANIE - Filtr Bayera (2x2)
# ============================================

#Wczytanie oryginalnego obrazu RGB
img_rgb = Image.open(image_path).convert("RGB")
image_rgb = np.array(img_rgb).astype(np.float32) / 255.0

# Dopasowanie rozmiaru obrazu do wielokrotności 2
H, W, C = image_rgb.shape
H_new = (H // 2) * 2
W_new = (W // 2) * 2
image_rgb = image_rgb[:H_new, :W_new, :]

print(f"Rozmiar obrazu: {image_rgb.shape}")

#   G R
#   B G

# Utworzenie 3 masek - (R, G, B)
mask_R = np.zeros((H_new, W_new), dtype=np.float32)
mask_G = np.zeros((H_new, W_new), dtype=np.float32)
mask_B = np.zeros((H_new, W_new), dtype=np.float32)

# Ustawiamy piksele zgodnie z wzorem Bayera
mask_R[0::2, 1::2] = 1  # R w prawym górnym rogu
mask_G[0::2, 0::2] = 1  # G w lewym górnym rogu
mask_G[1::2, 1::2] = 1  # G w prawym dolnym rogu
mask_B[1::2, 0::2] = 1  # B w lewym dolnym rogu

# Łączenie masek w filtr Bayera
bayer_filter = np.stack([mask_R, mask_G, mask_B], axis=-1)

# Symulacja odczytu z sensora kamery
sensor_image = image_rgb * bayer_filter

# Definicja kerneli do interpolacji brakujących pikseli
kernel_R = np.ones((2, 2), dtype=np.float32)       # suma = 4 
kernel_G = 0.5 * np.ones((2, 2), dtype=np.float32) # suma = 2 
kernel_B = np.ones((2, 2), dtype=np.float32)       # suma = 4

# interpolacja brakujących pikseli dla każdego kanału
R_interp = convolve2d(sensor_image[:, :, 0], kernel_R, padding_mode="constant")
G_interp = convolve2d(sensor_image[:, :, 1], kernel_G, padding_mode="constant")
B_interp = convolve2d(sensor_image[:, :, 2], kernel_B, padding_mode="constant")

# Łączenie z oryginalnymi wartościami z sensora    
R_final = np.where(mask_R == 1, sensor_image[:, :, 0], R_interp)
G_final = np.where(mask_G == 1, sensor_image[:, :, 1], G_interp)
B_final = np.where(mask_B == 1, sensor_image[:, :, 2], B_interp)

# Łączenie kanałów z powrotem w obraz RGB
reconstructed_image = np.stack([R_final, G_final, B_final], axis=-1)

# Przycinanie jednego piksela z dołu (z powodu paddingu 'constant' w konwolucji)
reconstructed_image = reconstructed_image[:-1, :, :]

# Przycinanie wartości do zakresu [0, 1]
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Konwersja do formatu uint8 do wyświetlenia
reconstructed_uint8 = (reconstructed_image * 255).astype(np.uint8)

plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.title("Oryginał")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Odczyt z sensora - filtr Bayera")
plt.imshow(sensor_image)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Demozaikowanie - filtr Bayera")
plt.imshow(reconstructed_uint8)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Demozaikowanie zakończone!")
print(f"Rozmiar wyjściowy: {reconstructed_uint8.shape}")


# ============================================
# Demozaikowanie - Fuji X-Trans (6x6)
# ============================================

H, W, C = image_rgb.shape  # rozmiar oryginalnego obrazu

# Definiowanie masek Fuji X-Trans 6x6
mask_R = np.array([[0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.float32)

mask_G = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]], dtype=np.float32)

mask_B = np.array([[0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=np.float32)

# Powielenie maski 6x6 na cały obraz
mask_R = np.tile(mask_R, (H // 6 + 1, W // 6 + 1))[:H, :W]
mask_G = np.tile(mask_G, (H // 6 + 1, W // 6 + 1))[:H, :W]
mask_B = np.tile(mask_B, (H // 6 + 1, W // 6 + 1))[:H, :W]
fuji_filter = np.stack([mask_R, mask_G, mask_B], axis=-1)

# Symulacja odczytu z sensora
sensor_image = image_rgb * fuji_filter

# Definicja kerneli do interpolacji brakujących pikseli
# Kernel 5x5 dla czerwonego kanału
kernel_R = np.ones((5, 5), dtype=np.float32)

# Kernel 3x3 dla zielonego kanału
kernel_G = np.ones((3, 3), dtype=np.float32)

# Kernel 5x5 dla niebieskiego kanału
kernel_B = np.ones((5, 5), dtype=np.float32)

# Kanał czerwony
R_sum = convolve2d(mask_R, kernel_R, padding_mode="constant")
R_interp = convolve2d(sensor_image[:, :, 0], kernel_R, padding_mode="constant") / (R_sum + 1e-6)
R_final = np.where(mask_R == 1, sensor_image[:, :, 0], R_interp)

# Kanał zielony
G_sum = convolve2d(mask_G, kernel_G, padding_mode="constant")
G_interp = convolve2d(sensor_image[:, :, 1], kernel_G, padding_mode="constant") / (G_sum + 1e-6)
G_final = np.where(mask_G == 1, sensor_image[:, :, 1], G_interp)

# Kanał niebieski
B_sum = convolve2d(mask_B, kernel_B, padding_mode="constant")
B_interp = convolve2d(sensor_image[:, :, 2], kernel_B, padding_mode="constant") / (B_sum + 1e-6)
B_final = np.where(mask_B == 1, sensor_image[:, :, 2], B_interp)

# Składanie kanałów z powrotem w obraz RGB
reconstructed_fuji = np.stack([R_final, G_final, B_final], axis=-1)
reconstructed_fuji = np.clip(reconstructed_fuji, 0, 1)
reconstructed_uint8 = (reconstructed_fuji * 255).astype(np.uint8)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Oryginał")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Odczyt z sensora - filtr Fuji X-Trans")
plt.imshow(sensor_image)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Demozaikowanie - filtr Fuji X-Trans")
plt.imshow(reconstructed_uint8)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Demozaikowanie Fuji X-Trans zakończone!")
print(f"Rozmiar wyjściowy: {reconstructed_uint8.shape}")

# ============================================
# Porównanie dwóch metod demozaikowania
# ============================================

plt.figure(figsize=(22, 12))

# Filtr Bayera
H, W, C = image_rgb.shape
H_new = (H // 2) * 2
W_new = (W // 2) * 2
image_rgb_bayer = image_rgb[:H_new, :W_new, :]

bayer_R = np.zeros((H_new, W_new), dtype=np.float32)
bayer_G = np.zeros((H_new, W_new), dtype=np.float32)
bayer_B = np.zeros((H_new, W_new), dtype=np.float32)

bayer_R[0::2, 1::2] = 1
bayer_G[0::2, 0::2] = 1
bayer_G[1::2, 1::2] = 1
bayer_B[1::2, 0::2] = 1

bayer_filter = np.stack([bayer_R, bayer_G, bayer_B], axis=-1)
sensor_image_bayer = image_rgb_bayer * bayer_filter

kernel_R = np.ones((2, 2), dtype=np.float32)
kernel_G = 0.5 * np.ones((2, 2), dtype=np.float32)
kernel_B = np.ones((2, 2), dtype=np.float32)

R_interp = convolve2d(sensor_image_bayer[:, :, 0], kernel_R, padding_mode="constant")
G_interp = convolve2d(sensor_image_bayer[:, :, 1], kernel_G, padding_mode="constant")
B_interp = convolve2d(sensor_image_bayer[:, :, 2], kernel_B, padding_mode="constant")

R_final = np.where(bayer_R == 1, sensor_image_bayer[:, :, 0], R_interp)
G_final = np.where(bayer_G == 1, sensor_image_bayer[:, :, 1], G_interp)
B_final = np.where(bayer_B == 1, sensor_image_bayer[:, :, 2], B_interp)

reconstructed_bayer = np.stack([R_final, G_final, B_final], axis=-1)
reconstructed_bayer = reconstructed_bayer[:-1, :, :]
reconstructed_bayer = np.clip(reconstructed_bayer, 0, 1)
reconstructed_bayer_uint8 = (reconstructed_bayer * 255).astype(np.uint8)

plt.subplot(2, 3, 1)
plt.title("Oryginał (Bayer)")
plt.imshow(image_rgb_bayer)
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Sensor - filtr Bayera")
plt.imshow(sensor_image_bayer)
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Demozaikowanie - filtr Bayera")
plt.imshow(reconstructed_bayer_uint8)
plt.axis("off")

# Filtr Fuji X-Trans
H, W, C = image_rgb.shape

mask_R = np.array([[0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.float32)
mask_G = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]], dtype=np.float32)
mask_B = np.array([[0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=np.float32)

mask_R = np.tile(mask_R, (H // 6 + 1, W // 6 + 1))[:H, :W]
mask_G = np.tile(mask_G, (H // 6 + 1, W // 6 + 1))[:H, :W]
mask_B = np.tile(mask_B, (H // 6 + 1, W // 6 + 1))[:H, :W]

fuji_filter = np.stack([mask_R, mask_G, mask_B], axis=-1)
sensor_image_fuji = image_rgb * fuji_filter

kernel_G = np.ones((3, 3), dtype=np.float32)
kernel_R = np.ones((5, 5), dtype=np.float32)
kernel_B = np.ones((5, 5), dtype=np.float32)

R_sum = convolve2d(mask_R, kernel_R, padding_mode="constant")
R_interp = convolve2d(sensor_image_fuji[:, :, 0], kernel_R, padding_mode="constant") / (R_sum + 1e-6)
R_final = np.where(mask_R == 1, sensor_image_fuji[:, :, 0], R_interp)

G_sum = convolve2d(mask_G, kernel_G, padding_mode="constant")
G_interp = convolve2d(sensor_image_fuji[:, :, 1], kernel_G, padding_mode="constant") / (G_sum + 1e-6)
G_final = np.where(mask_G == 1, sensor_image_fuji[:, :, 1], G_interp)

B_sum = convolve2d(mask_B, kernel_B, padding_mode="constant")
B_interp = convolve2d(sensor_image_fuji[:, :, 2], kernel_B, padding_mode="constant") / (B_sum + 1e-6)
B_final = np.where(mask_B == 1, sensor_image_fuji[:, :, 2], B_interp)

reconstructed_fuji = np.stack([R_final, G_final, B_final], axis=-1)
reconstructed_fuji = np.clip(reconstructed_fuji, 0, 1)
reconstructed_fuji_uint8 = (reconstructed_fuji * 255).astype(np.uint8)

plt.subplot(2, 3, 4)
plt.title("Oryginał (Fuji X-Trans)")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Sensor - filtr Fuji X-Trans")
plt.imshow(sensor_image_fuji)
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Demozaikowanie - filtr Fuji X-Trans")
plt.imshow(reconstructed_fuji_uint8)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Porównanie demozaikowania Bayera vs Fuji X-Trans zakończone!")
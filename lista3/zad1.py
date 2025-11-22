import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy.random as rnd
from skimage import io
from PIL import Image

def f1(x):
  return np.sin(x)

def f2(x):
  x = np.asarray(x)
  result = np.zeros_like(x, dtype=float)
  mask = x != 0
  result[mask] = np.sin(1 / x[mask])
  return result

def f3(x):
  return np.sign(np.sin(8 * x))

def h1(t):
  return (t >= 0) & (t <= 1)

def h2(t):
  return (t >= -0.5) & (t <= 0.5)

def h3(t):
  mask = (t >= -1) & (t <= 1)
  result = np.zeros_like(t)
  result[mask] = 1 - np.abs(t[mask])
  return result

def interpolate(x_old, y_old, x_new, kernel):
  y_new = np.zeros_like(x_new, dtype=float)
  dis = x_old[1] - x_old[0]
  for i, x in enumerate(x_new):
    weights = kernel((x - x_old) / dis).astype(float)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
      weights /= weight_sum
      y_new[i] = np.sum(y_old * weights)
    else:
      nearest_idx = np.argmin(np.abs(x - x_old))
      y_new[i] = y_old[nearest_idx]
  return y_new

N = 100
x_original = np.linspace(-np.pi, np.pi, N)

functions = [
    ("sin(x)", f1),
    ("sin(1/x)", f2),
    ("sgn(sin(8x))", f3)
]

kernels = [
    ("Prostokątne", h1),
    ("Sąsiadujące", h2),
    ("Trójkątne", h3),
]

scales = [2, 4, 10]

for scale in scales:
  plt.figure(figsize=(12, 8))
  plt.suptitle(f"Interpolacja funkcji - Skala {scale}x", fontsize=16)
  
  print(f"\nSkala {scale}x:")
  
  i = 1
  for func_name, func in functions:
    plt.subplot(2, 2, i)
    
    y_original = func(x_original)
    plt.plot(x_original, y_original, "ko-", markersize=3, linewidth=1, label="Oryginał", alpha=0.7)
    
    x_new = np.linspace(-np.pi, np.pi, N * scale)
    y_true = func(x_new)
    
    print(f"  {func_name}:")
    
    for kernel_name, kernel in kernels:
      y_interp = interpolate(x_original, y_original, x_new, kernel)
      mse = mean_squared_error(y_true, y_interp)
      plt.plot(x_new, y_interp, "--", linewidth=1.5, 
           label=f"{kernel_name} (MSE={mse:.4f})")
      print(f"    {kernel_name}: MSE = {mse:.4f}")
    
    print()
    
    plt.title(f"{func_name}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    i += 1
  
  plt.tight_layout()
  plt.show()

print("\n--- DODATKOWE ZADANIE 4 ---")

rng = np.random.default_rng()

def interpolate_median_dist(x_old, y_old, x_new, kernel):
  y_new = np.zeros_like(x_new, dtype=float)

  diffs = np.diff(np.sort(x_old))
  if len(diffs) > 0:
      dis = np.median(diffs)
  else:
      dis = 1.0

  for i, x in enumerate(x_new):
    weights = kernel((x - x_old) / dis).astype(float)
    weight_sum = np.sum(weights)
    if weight_sum > 0:
      weights /= weight_sum
      y_new[i] = np.sum(y_old * weights)
    else:
      nearest_idx = np.argmin(np.abs(x - x_old))
      y_new[i] = y_old[nearest_idx]
  return y_new


for func_name, func in functions:
  print(f"\nFunkcja: {func_name}")
  for kernel_name, kernel in kernels:
    print(f"  Jądro: {kernel_name}")
    
    x_evently = np.linspace(-np.pi, np.pi, N)
    y_evently = func(x_evently)
    
    x_random = rng.normal(0, np.pi/2, size=N)
    x_random = np.clip(x_random, -np.pi, np.pi)
    x_random = np.sort(x_random)
    y_random = func(x_random)
    
    x_new = np.linspace(-np.pi, np.pi, N * scales[2])
    y_true = func(x_new)

    y_interp_evently = interpolate_median_dist(x_evently, y_evently, x_new, kernel)
    mse_evently = mean_squared_error(y_true, y_interp_evently)
    
    y_interp_random = interpolate_median_dist(x_random, y_random, x_new, kernel)
    mse_random = mean_squared_error(y_true, y_interp_random)
    
    print(f"    MSE (równomierne): {mse_evently:.4f}")
    print(f"    MSE (losowe):      {mse_random:.4f}")
    print(f"    Różnica:           {abs(mse_evently - mse_random):.4f}")

print("\n--- KONIEC DODATKOWEGO ZADANIA 4 ---")


print("\n--- DODATKOWE ZADANIE 5 ---")

target = 16
x_target = np.linspace(-np.pi, np.pi, N * target)

for func_name, func in functions:
  print(f"\nFunkcja: {func_name}")
  for kernel_name, kernel in kernels:
    y_single = interpolate(x_original, func(x_original), x_target, kernel)
    mse_single = mean_squared_error(func(x_target), y_single)
    
    x_curr = x_original.copy()
    y_curr = func(x_curr).copy()
    
    for i in range(4):
      x_next = np.linspace(-np.pi, np.pi, len(x_curr) * 2)
      y_next = interpolate(x_curr, y_curr, x_next, kernel)
      x_curr = x_next
      y_curr = y_next
    
    y_multi = y_curr
    if len(y_multi) != len(x_target):
      y_multi = interpolate(x_curr, y_curr, x_target, kernel)
    mse_multi = mean_squared_error(func(x_target), y_multi)

    print(f"  {kernel_name}: MSE pojedynczy 16x = {mse_single:.4f}, MSE wieloetapowy 4×(2x) = {mse_multi:.4f}")

print("\n--- KONIEC DODATKOWEGO ZADANIA 5 ---")

# ZADANIE 2

def downscale_average(image, s):
    H, W = image.shape
    k = s
    p = 0
    H_out = (H - k + 1 + p) // s
    W_out = (W - k + 1 + p) // s
    out = np.zeros((H_out, W_out))
    for i_out in range(H_out):
        for j_out in range(W_out):
            i = i_out * s
            j = j_out * s
            patch = image[i:i+k, j:j+k]
            out[i_out, j_out] = patch.mean()
    return out

def upscale_image(image, s, kernel):
    H, W = image.shape
    x_old = np.arange(W)
    y_old = np.arange(H)  

    x_new = np.linspace(0, W - 1, W * s)
    y_new = np.linspace(0, H - 1, H * s)

    temp = np.zeros((H, W * s), dtype=float)
    for r in range(H):
        temp[r, :] = interpolate(x_old, image[r, :], x_new, kernel)

    upscaled = np.zeros((H * s, W * s), dtype=float)
    for c in range(W * s):
      upscaled[:, c] = interpolate(y_old, temp[:, c], y_new, kernel)

    return upscaled

def downscale_maxpool(image, s):
    H, W = image.shape
    k = s
    p = 0
    H_out = (H - k + 1 + p) // s
    W_out = (W - k + 1 + p) // s
    out = np.zeros((H_out, W_out))
    for i_out in range(H_out):
        for j_out in range(W_out):
            i = i_out * s
            j = j_out * s
            patch = image[i:i+k, j:j+k]
            out[i_out, j_out] = patch.max()
    return out
  
image_path = "lista2/pug.jpg"
s = 4;
kernel = h2

img = Image.open(image_path).convert("L")
image = np.array(img)

downscaled = downscale_average(image, s)
downscaled2 = downscale_maxpool(image, s)
upscaled = upscale_image(downscaled, s, kernel)
upscaled2 = upscale_image(downscaled2, s, kernel)
upscaled3 = upscale_image(image, s, kernel)

H, W = upscaled.shape
image_cropped = image[:H, :W]

H_down, W_down = downscaled.shape
image_cut = image[:H_down * s, :W_down * s]

subsample = image_cut[::s, ::s]

mse_down = mean_squared_error(subsample.astype(float).ravel(), downscaled.astype(float).ravel())
print(f"MSE pomniejszenia: {mse_down:.4f}")

mse = mean_squared_error(image_cropped, upscaled)
print(f"MSE po pomniejszeniu i powiększeniu: {mse:.4f}")

mse_down2 = mean_squared_error(subsample.astype(float).ravel(), downscaled2.astype(float).ravel())
print(f"MSE pomniejszenia max-pooling: {mse_down2:.4f}")

mse2 = mean_squared_error(image_cropped, upscaled2)
print(f"MSE po pomniejszeniu max-pooling i powiększeniu: {mse2:.4f}")

upscaled3_cropped = upscaled3[:image_cropped.shape[0], :image_cropped.shape[1]]
mse3 = mean_squared_error(image_cropped.astype(float).ravel(), upscaled3_cropped.ravel())
print(f"MSE po powiększeniu bez pomniejszania: {mse3:.4f}")

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title(f"Oryginał ({image.shape[1]}x{image.shape[0]})")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(downscaled, cmap='gray', interpolation='nearest')
plt.title(f"Pomniejszony {s} razy ({downscaled.shape[1]}x{downscaled.shape[0]}) \nMSE = {mse_down:.4f}")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(upscaled, cmap='gray', interpolation='nearest')
plt.title(f"Powiększony {s} razy ({upscaled.shape[1]}x{upscaled.shape[0]}) \nMSE = {mse:.4f}")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(downscaled2, cmap='gray', interpolation='nearest')
plt.title(f"Pomniejszony max-pooling {s} razy ({downscaled2.shape[1]}x{downscaled2.shape[0]}) \nMSE = {mse_down2:.4f}")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(upscaled2, cmap='gray', interpolation='nearest')
plt.title(f"Powiększony max-pooling {s} razy ({upscaled2.shape[1]}x{upscaled2.shape[0]}) \nMSE = {mse2:.4f}")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(upscaled3, cmap='gray', interpolation='nearest')
plt.title(f"Powiększony bez pomniejszania ({upscaled3.shape[1]}x{upscaled3.shape[0]}) \nMSE = {mse3:.4f}")
plt.axis('off')

plt.tight_layout()
plt.show()

def downscale_average_rgb(image_rgb, s):
    channels = []
    for ch in range(3):
        ch_img = image_rgb[..., ch]
        ch_down = downscale_average(ch_img, s)
        channels.append(ch_down)
    return np.stack(channels, axis=-1)

def downscale_maxpool_rgb(image_rgb, s):
    channels = []
    for ch in range(3):
        ch_img = image_rgb[..., ch]
        ch_down = downscale_maxpool(ch_img, s)
        channels.append(ch_down)
    return np.stack(channels, axis=-1)

def upscale_image_rgb(image_rgb_small, s, kernel):
    channels = []
    for ch in range(3):
        ch_img = image_rgb_small[..., ch]
        ch_up = upscale_image(ch_img.astype(float), s, kernel)
        channels.append(ch_up)
    return np.stack(channels, axis=-1)


def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
  
image__path = "lista2/pug.jpg"
s = 4;
kernel = h2
img = Image.open(image_path).convert("RGB")
image = np.array(img)

downscaled_rgb = downscale_average_rgb(image, s)
downscaled_rgb2 = downscale_maxpool_rgb(image, s)
upscaled_rgb = upscale_image_rgb(downscaled_rgb, s, kernel)
upscaled_rgb2 = upscale_image_rgb(downscaled_rgb2, s, kernel)
upscaled_rgb3 = upscale_image_rgb(image, s, kernel)

H, W = upscaled_rgb.shape[:2]
image_cropped = image[:H, :W].astype(float)

H_down, W_down = downscaled.shape
image_cut = image[:H_down * s, :W_down * s]

subsample_rgb = image_cut[::s, ::s]

mse_down = mean_squared_error(subsample_rgb.astype(float).ravel(), downscaled_rgb.astype(float).ravel())
print(f"MSE pomniejszenia: {mse_down:.4f}")

mse = mean_squared_error(image_cropped.ravel(), upscaled_rgb.ravel())
print(f"MSE po pomniejszeniu i powiększeniu: {mse:.4f}")

mse_down2 = mean_squared_error(subsample_rgb.astype(float).ravel(), downscaled_rgb2.astype(float).ravel())
print(f"MSE pomniejszenia max-pooling: {mse_down2:.4f}")

mse2 = mean_squared_error(image_cropped.ravel(), upscaled_rgb2.ravel())
print(f"MSE po pomniejszeniu max-pooling i powiększeniu: {mse2:.4f}")

upscaled3_cropped = upscaled_rgb3[:image_cropped.shape[0], :image_cropped.shape[1], :]
mse3 = mean_squared_error(image_cropped.ravel(), upscaled3_cropped.ravel())
print(f"MSE po powiększeniu bez pomniejszania: {mse3:.4f}")

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(image)
plt.title(f"Oryginał ({image.shape[1]}x{image.shape[0]})")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(downscaled_rgb.astype(np.uint8), interpolation='nearest')
plt.title(f"Pomniejszony {s} razy ({downscaled_rgb.shape[1]}x{downscaled_rgb.shape[0]}) \nMSE = {mse_down:.4f}")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(to_uint8(upscaled_rgb), interpolation='nearest')
plt.title(f"Powiększony {s} razy ({upscaled_rgb.shape[1]}x{upscaled_rgb.shape[0]}) \nMSE = {mse:.4f}")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(downscaled_rgb2.astype(np.uint8), interpolation='nearest')
plt.title(f"Pomniejszony max-pooling {s} razy ({downscaled_rgb2.shape[1]}x{downscaled_rgb2.shape[0]}) \nMSE = {mse_down2:.4f}")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(to_uint8(upscaled_rgb2), interpolation='nearest')
plt.title(f"Powiększony max-pooling {s} razy ({upscaled_rgb2.shape[1]}x{upscaled_rgb2.shape[0]}) \nMSE = {mse2:.4f}")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(to_uint8(upscaled_rgb3), interpolation='nearest')
plt.title(f"Powiększony bez pomniejszania ({upscaled_rgb3.shape[1]}x{upscaled_rgb3.shape[0]}) \nMSE = {mse3:.4f}")
plt.axis('off')

plt.tight_layout()
plt.show()

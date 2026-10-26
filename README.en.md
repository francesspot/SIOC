[Polski](README.md) | **English**

# SIOC — Lab Assignments

Solutions to the lab assignment lists for the SIOC course, written in Python. The topics move from Python basics through plotting to signal and image processing implemented by hand with NumPy.

| Folder | Topics |
|--------|--------|
| `lista1` | Python warm-up: counting unique letters, double letters and digit sums in text; finding zeros of a function numerically |
| `lista2` | NumPy and Matplotlib: 2D plots and subplots, 3D surface and contour plots, displaying an image, generating a matrix with a circle, computing the arc length of a curve |
| `lista3` | **Interpolation** with different kernels (nearest-neighbour, linear, etc.) and MSE comparison; uniform vs random sample points; single vs multi-step interpolation; **image scaling** — downscaling (averaging, max-pooling) and upscaling, in grayscale and RGB |
| `lista4` | **2D convolution** implemented from scratch; edge detection (Sobel, Laplace, Scharr, Prewitt); blurring (Gaussian, box); sharpening and unsharp masking; **demosaicing** of a Bayer-filter image |
| `lista5` | Jupyter notebook: removing periodic noise from an image using the **2D Fourier transform** |

## Requirements

Python 3 with:

```bash
pip install numpy matplotlib scipy scikit-image scikit-learn pillow jupyter
```

## Running

Run the scripts **from the repository root**, because image paths are relative to it (e.g. `lista3/pug.jpg`):

```bash
python lista4/zad1.py
```

Open `lista5/task.ipynb` in Jupyter or VS Code. It downloads the test image from the internet, so it needs a network connection.

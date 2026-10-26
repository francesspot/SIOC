🇵🇱 [Polski](README.md) | 🇬🇧 **English**

# 📡 Digital Signals and Images — Labs

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)

> 🎓 Lab assignments for the **Digital Signals and Images** course (*Sygnały i Obrazy Cyfrowe*, Wrocław University of Science and Technology)

Signal and image processing from the inside 🔬 — instead of calling ready-made library functions, I wrote most of the algorithms (interpolation, convolution, filters, demosaicing) myself in NumPy.

## 🗂️ What's inside

| | Topic |
|---|---|
| 🐍 `lista1` | Python warm-up: text operations, finding zeros of a function numerically |
| 📈 `lista2` | 2D and 3D plots, treating an image as a matrix, curve length |
| 🔍 `lista3` | **Interpolation** with different kernels and error comparison (MSE); **image scaling** — downscaling (averaging, max-pooling) and upscaling, grayscale and RGB |
| 🖼️ `lista4` | **2D convolution from scratch**; edge detection (Sobel, Laplace, Scharr, Prewitt); blur, sharpening, unsharp masking; **demosaicing** a Bayer-filter image |
| 🌊 `lista5` | Removing noise from an image with the **2D Fourier transform** |

## 🛠️ Tech

Python · NumPy · Matplotlib · SciPy · scikit-image · Pillow · Jupyter

## 🚀 Run it

```bash
pip install numpy matplotlib scipy scikit-image scikit-learn pillow jupyter
python lista4/zad1.py      # run from the repository root
```

**Polski** | [English](README.en.md)

# SIOC — listy zadań

Rozwiązania list zadań laboratoryjnych z kursu SIOC, napisane w Pythonie. Tematy prowadzą od podstaw Pythona, przez wykresy, do przetwarzania sygnałów i obrazów zaimplementowanego samodzielnie w NumPy.

| Folder | Zakres |
|--------|--------|
| `lista1` | Rozgrzewka w Pythonie: liczenie unikalnych liter, podwójnych liter i sumy cyfr w tekście; numeryczne szukanie miejsc zerowych funkcji |
| `lista2` | NumPy i Matplotlib: wykresy 2D i podwykresy, powierzchnia 3D i wykres konturowy, wyświetlanie obrazu, generowanie macierzy z okręgiem, długość krzywej |
| `lista3` | **Interpolacja** różnymi jądrami (najbliższy sąsiad, liniowa itd.) i porównanie MSE; punkty rozmieszczone równomiernie i losowo; interpolacja jednoetapowa i wieloetapowa; **skalowanie obrazów** — zmniejszanie (uśrednianie, max-pooling) i powiększanie, w skali szarości i RGB |
| `lista4` | **Splot 2D** zaimplementowany od podstaw; wykrywanie krawędzi (Sobel, Laplace, Scharr, Prewitt); rozmycie (Gaussa, pudełkowe); wyostrzanie i maskowanie nieostrości; **demozaikowanie** obrazu z filtrem Bayera |
| `lista5` | Notatnik Jupyter: usuwanie okresowego szumu z obrazu za pomocą **dwuwymiarowej transformaty Fouriera** |

## Wymagania

Python 3 oraz:

```bash
pip install numpy matplotlib scipy scikit-image scikit-learn pillow jupyter
```

## Uruchamianie

Skrypty uruchamiaj **z katalogu głównego repozytorium**, bo ścieżki do obrazów są względne (np. `lista3/pug.jpg`):

```bash
python lista4/zad1.py
```

`lista5/task.ipynb` otwórz w Jupyterze lub VS Code. Notatnik pobiera obraz testowy z internetu, więc potrzebne jest połączenie z siecią.

import math


# Zadanie 1.2

def unique_letters(text: str) -> int:
  return len({ch.lower() for ch in text if ch.isalpha()})

def double_letter_count(text: str) -> int:
  count = 0
  text = text.lower()
  for i in range(len(text) - 1):
    if text[i].isalpha() and text[i] == text[i + 1]:
      count += 1
  return count

def sum_of_digits(text: str) -> int:
  return sum(int(ch) for ch in text if ch.isdigit())


# Zadanie 1.3

def find_zeros(f, start, end, step=0.01):
  zeros = []
  x = start
  while x < end:
    if abs(f(x)) < step:
      zeros.append(round(x, 2))
    x += step
  return zeros
import numpy as np
import trimesh
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from pyquaternion import Quaternion
import math
import random


def rotation_matrix(axis, angle):
  '''
  Macierz rotacji dla obrotu danego przez oś i kąt
  '''
  [a, b, c, d] = Quaternion(axis=axis, angle=angle)

  R = np.array([[a*a+b*b-c*c-d*d, 
                 2 * (b*c-a*d), 
                 2 * (b*d+a*c)],
                [2 * (b*c+a*d), 
                 a*a+c*c-b*b-d*d, 
                 2 * (c*d-a*b)],
                [2 * (b*d-a*c), 
                 2 * (c*d+a*b), 
                 a*a+d*d-b*b-c*c]])
  return R


def transform_matrix(A, B):
  '''
  Funkcja znajdująca najlepszą transformację między
  chmurami punktów A i B metodą najmniejszych kwadratów
  Wejście:
    A - macierz m x n
    B - macierz m x n
  Wyjście:
    T - macierz transformacji (n+1) x (n+1) 
  '''
  n = A.shape[1]

  # Wyznaczenie centroidów
  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)
  # Utworzenie nowych macierzy A1 i B1 przez
  # przedstawienie każdego punktu A i B jako
  # jego odległości od centroidu
  A1 = A - centroid_A
  B1 = B - centroid_B

  H = np.dot(A1.T, B1)
  # Rozkład H według wartości osobliwych
  U, D, Vt = np.linalg.svd(H)
  # Wyznaczenie możliwej macierzy obrotu
  R = np.dot(Vt.T, U.T)

  # Obsługa szczególnego przypadku odbicia
  if np.linalg.det(R) < 0:
    Vt[n - 1, :] *= -1
    R = np.dot(Vt.T, U.T)

  # Wektor przesunięcia jako różnica centroidów
  t = centroid_B.T - np.dot(R, centroid_A.T)

  # Konkatenacja macierzy rotacji i wektora
  # przesunięcia w celu uzyskania ostatecznej
  # transformacji
  T = np.identity(n + 1)
  T[:n, :n] = R
  T[:n, n] = t

  return T


def calculate_distance(A, B):
  '''
  Funkcja znajdująca najbliższego sąsiada wśród B 
  dla każdego punktu należącego do A w metryce
  euklidesowej
  Wejście:
    A - macierz m x n
    B - macierz m x n
  Wyjście:
    distances - odległości od najbliższego sąsiada
    indices - indeksy najbliższego sąsiada
  '''
  nn = NearestNeighbors(n_neighbors=1)
  nn.fit(B)
  distances, indices = nn.kneighbors(A, return_distance=True)
  return distances.ravel(), indices.ravel()


def icp(A, B, max_iterations, tolerance=0.001):
  '''
  Algorytm Iterative Closest Point iteracyjnie znajdujący 
  najlepszą transformację między chmurami punktów A i B
  Wejście:
    A - macierz m x n
    B - macierz m x n
  Wyjście:
    T - macierz transformacji
  '''
  n = A.shape[1]

  A1 = np.ones((n + 1, A.shape[0]))
  B1 = np.ones((n + 1, B.shape[0]))
  A1[:n, :] = np.copy(A.T)
  B1[:n, :] = np.copy(B.T)
  
  prev_error = 0
  for i in range(iterations):
    # Znalezienie najbliższego sąsiada między A1 a B1
    distances, indices = calculate_distance(A1[:n, :].T, B1[:n, :].T)

    # Obliczenie transformacji między A1 a najbliższymi punktami B1
    T = transform_matrix(A1[:n, :].T, B1[:n, indices].T)
    
    # Przesunięcie A1 o wyznaczoną transformację
    A1 = np.dot(T, A1)

    # Sprawdzenie błędu
    mean_error = np.mean(distances)
    if abs(prev_error - mean_error) < tolerance:
      break

    prev_error = mean_error

  # Wyznaczenie ostatecznej transormacji
  T = transform_matrix(A, A1[:n, :].T)
  return T


mesh = trimesh.load('bunny.stl')
density = 3000
dim = 3
translation = 0.5
rotation = 0.5
scaler = MinMaxScaler(feature_range=(0, 1))

A = mesh.sample(density)
A = scaler.fit_transform(A)

B = mesh.sample(density)
B = scaler.fit_transform(B)

t = np.random.rand(dim) * translation
B += t

R = rotation_matrix(np.random.rand(dim), 
                    np.random.rand() * rotation)

iterations = 15
for i in range(iterations):
  T = icp(B, A, iterations + 1)

print(T)
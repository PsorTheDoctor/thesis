def transform_matrix(A, B):
  n = A.shape[1]
  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)

  A1 = A - centroid_A
  B1 = B - centroid_B

  H = np.dot(A1.T, B1)
  U, D, Vt = np.linalg.svd(H)
  R = np.dot(Vt.T, U.T)

  if np.linalg.det(R) < 0:
    Vt[n - 1, :] *= -1
    R = np.dot(Vt.T, U.T)

  t = centroid_B.T - np.dot(R, centroid_A.T)
  return R, t
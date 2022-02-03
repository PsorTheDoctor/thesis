def icp(A, B, max_iterations, tolerance=0.001):

  n = A.shape[1]
  A1 = np.ones((n + 1, A.shape[0]))
  B1 = np.ones((n + 1, B.shape[0]))
  A1[:n, :] = np.copy(A.T)
  B1[:n, :] = np.copy(B.T)
  
  prev_error = 0
  for i in range(iterations):
    distances, indices = calculate_distance(A1[:n, :].T,
                                            B1[:n, :].T)
    T = transform_matrix(A1[:n, :].T, B1[:n, indices].T)
    A1 = np.dot(T, A1)

    mean_error = np.mean(distances)
    if abs(prev_error - mean_error) < tolerance:
      break
    prev_error = mean_error

  T = transform_matrix(A, A1[:n, :].T)
  return T


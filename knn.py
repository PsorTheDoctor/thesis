def knn(points, p, k=1):
  distances = []
  for idx in points:
    for coord in points[idx]:

      dist = math.sqrt((coord[0] - p[0])**2 
                        + (coord[1] - p[1])**2)
      distances.append((dist, idx))

  distances = sorted(distances)[:k]
  return distances
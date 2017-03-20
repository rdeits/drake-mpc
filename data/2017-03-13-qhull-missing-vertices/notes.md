We ran into some issues with QHull on this particular dataset. When QHull is provided with the given A and b matrices (as a polyhedron A x <= b) and the given interior point, it gives the following error:

```
While executing:  | qhalf Fp
Options selected for Qhull 2012.1 2012/02/18:
  run-id 1855596596  Halfspace  Fpoint-intersect
QH6072 qhull input error: coordinates for feasible point do not finish out the line: 0.17504703998565674
```

CDDLib (in float and exact mode) finds all 16384 vertices of the polytope. When running QHull from Julia, it also finds all the vertices. We think that's because the Julia interface is providing the origin as the interior point, rather than the one in interior_point.txt. We confirmed that by providing the origin as the interior point in the Python code, which allowed QHull to successfully find the vertices.

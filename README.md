Approximation of second order derivative by dividing nodal point space twice.
  - first by 2s+1 times [uniformally]
  - then by k times [uniformally]
 
** uniformallity is not required but simplifies computations.

---
##### API Docs

```python 
# 2ks+1 method approximate
def rhs(y, x):
    return 2*y**2/(1+x)

two_ks_method_approx(rhs=rhs, number_of_iterations=5, s=2, k=4)
```

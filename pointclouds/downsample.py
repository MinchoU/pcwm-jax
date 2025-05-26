import ninjax as nj
import jax
import jax.numpy as jnp
from .nn_utils import farthest_point_sample

class FPS(nj.Module):
    def __init__(self, num_points, **kw):
        self.num_points = num_points
        for key, val in kw.items():
            setattr(self, key, val)

    def __call__(self, points, mode):
        return farthest_point_sample(points, self.num_points, True)[1]

class MultiFPS(nj.Module):
  def __init__(self, num_points, **kw):
    super().__init__()
    self.num_points = tuple(num_points)
    for k,v in kw.items(): setattr(self, k, v)

  def __call__(self, points, mode):
    @jax.jit
    def multi_fps(pts):
      points = []
      for n in self.num_points:
        _, point = farthest_point_sample(pts, n, True)
        points.append(point)
      # shape = (M, N) M=len(num_points), N=pts.shape[0]
      return jnp.concatenate(points, axis=-2)

    return multi_fps(points)
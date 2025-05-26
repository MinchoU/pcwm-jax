import jax
import jax.numpy as jnp
from functools import partial
import jax.ad_checkpoint as adc
import functools
from typing import Callable
from dreamerv3.nets import Initializer
import dreamerv3.ninjax as nj
import dreamerv3.jaxutils as jaxutils
# from latest dreamerv3
sg = jax.lax.stop_gradient
f32 = jnp.float32
COMPUTE_DTYPE = jnp.bfloat16
# COMPUTE_DTYPE = jaxutils.COMPUTE_DTYPE

@partial(jax.custom_vjp, nondiff_argnums=[1, 2])
def ensure_dtypes(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert x.dtype == fwd, (x.dtype, fwd)
  return x
def ensure_dtypes_fwd(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  return ensure_dtypes(x, fwd, bwd), ()
def ensure_dtypes_bwd(fwd, bwd, cache, dx):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert dx.dtype == bwd, (dx.dtype, bwd)
  return (dx,)
ensure_dtypes.defvjp(ensure_dtypes_fwd, ensure_dtypes_bwd)

def cast(xs, force=False):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: COMPUTE_DTYPE(x) if should(x) else x, xs)

def init(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    src_squared = jnp.sum(src ** 2, -1)
    dst_squared = jnp.sum(dst ** 2, -1)
    dist = src_squared[:, :, None] - 2 * jnp.matmul(src, jnp.transpose(dst, (0, 2, 1))) + dst_squared[:, None, :]
    
    return sg(jnp.maximum(dist, 0.0))

@jax.jit
def index_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, S] 또는 [B, S, K]
    """
    if len(idx.shape) == 2:  # [B, S]
        idx_expanded = idx[..., None]  # [B, S, 1]
        return jnp.take_along_axis(points, idx_expanded, axis=1)  # [B, S, C]
    
    else:
        raise ValueError(f"index_points: idx must have ndim 2 or 3, got {idx.ndim}")

@jax.jit
def index_points_3d(features, indices):
    """
    Args:
        features: shape (B, N, C)
        indices: shape (B, npoint, nsample)
    
    Returns:
        shape (B, npoint, nsample, C)
    """
    B, N, C = features.shape
    _, S, K = indices.shape
    one_hot = jax.nn.one_hot(indices, num_classes=N, dtype=features.dtype)
    group_indices = jnp.einsum('bskn,bnc->bskc', one_hot, features)
    return group_indices

def farthest_point_sample(points, num_points, check_val=False):
    """
    input:
        points: pointcloud data, [B, N, C]
        num_points: number of samples
        check_val: if True, xyzw input
    Return

    To avoid cumbersome random seed, select the first point as the most center point.
    """
    def fps_one(p):
        if check_val:
            coords_before = p[:, :3]
            mask = p[:, 3] > 0
            coords_after = p[:, 4:]
            
            if coords_after.shape[1] > 0:
                coords = jnp.concatenate([coords_before, coords_after], axis=1)
            else:
                coords = coords_before
        else:
            coords = p
            coords_before = p[:, :3]
            coords_after = p[:, 3:]
            mask = jnp.full(coords.shape[0], True, dtype=bool)

        centroid = jnp.mean(coords, axis=0)
        center_dists = jnp.sum((coords - centroid)**2, axis=1)
        first_idx = jnp.argmin(center_dists)

        N = coords.shape[0]
        centroids = jnp.zeros(num_points, jnp.int32)

        distances = jnp.full(N, -jnp.inf)
        distances = jnp.where(mask, jnp.full(N, jnp.inf), distances)
        centroids = centroids.at[0].set(first_idx)
        
        c = coords_before[first_idx]
        diff = coords_before - c
        dist2 = jnp.sum(diff * diff, axis=1)
        
        dist2 = jnp.where(mask, dist2, jnp.inf)
        distances = jnp.minimum(distances, dist2)
        
        selected_mask = jnp.zeros(N, dtype=bool).at[first_idx].set(True)
        distances = jnp.where(selected_mask, -jnp.inf, distances)
        
        def body_fn(i, state):
            centroids, distances = state
            next_idx = jnp.argmax(distances)
            centroids = centroids.at[i].set(next_idx)
            c = coords_before[next_idx]

            dist2 = jnp.sum((coords_before - c)**2, axis=1)
            dist2 = jnp.where(mask, dist2, jnp.inf)
            distances = jnp.minimum(distances, dist2)
            distances = distances.at[next_idx].set(-jnp.inf)
            return centroids, distances

        centroids, _ = jax.lax.fori_loop(1, num_points, body_fn,
                                         (centroids, distances))
        sampled_coords = coords[centroids]   # [num_points, C]
        # return sg(centroids), sg(sampled_coords)
        return sg(centroids), sampled_coords
        # return centroids, sampled_coords

    indices, coords = jax.vmap(fps_one)(points)
    return indices, coords

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = jax.lax.top_k(-sqrdists, nsample)
    return sg(group_idx)

@jax.jit
def sample_and_group_all(xyz, points, density_scale=None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
        density_scale: density scale, [B, N, 1]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
        grouped_xyz: grouped xyz, [B, 1, N, C]
        grouped_density: grouped density, [B, 1, N, 1]
    """
    # B, N, C = xyz.shape
    
    # Use mean as center
    new_xyz = jnp.mean(xyz, axis=1, keepdims=True)  # [B, 1, C]
    
    # Group all points
    grouped_xyz = jnp.expand_dims(xyz, 1) - jnp.expand_dims(new_xyz, 2)  # [B, 1, N, C]
    
    if points is not None:
        new_points = jnp.concatenate([grouped_xyz, jnp.expand_dims(points, 1)], axis=-1)  # [B, 1, N, C+D]
    else:
        new_points = grouped_xyz
    
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = jnp.expand_dims(density_scale, 1)  # [B, 1, N, 1]
        return new_xyz, new_points, grouped_xyz, grouped_density


def compute_density(xyz, bandwidth):
    """
    Compute density for points
    
    Input:
        xyz: input points position data, [B, N, C]
        bandwidth: bandwidth for density estimation
    Return:
        density: density for each point, [B, N]
    """
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussian_density = jnp.exp(-sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = jnp.mean(gaussian_density, axis=-1)
    
    return xyz_density

_fps_jit = partial(jax.jit, static_argnums=(1,2))(farthest_point_sample)

def _sample_and_group_jit(npoint, nsample, xyz, points, new_xyz=None, nn_idx=None, density_scale=None):
    # B, N, C = xyz.shape
    # S = npoint
    if new_xyz is None:
        fps_idx, _ = _fps_jit(xyz, npoint)             # [B, npoint]
        new_xyz = sg(index_points(xyz, fps_idx))           # [B, npoint, C]
        
    if nn_idx is None:
        idx = knn_point(nsample, xyz, new_xyz)         # [B, npoint, nsample] # do this in env step?
    else:
        idx = nn_idx

    grouped_xyz = sg(index_points_3d(xyz, idx))          # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None]

    if points is not None:
        # grouped_points = index_points(points, idx) # [B, npoint, nsample, D]
        grouped_points = index_points_3d(points, idx)  # [B, npoint, nsample, D]
        new_points = jnp.concatenate([grouped_xyz_norm, grouped_points], -1)
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx, None
    else:
        gd = index_points(density_scale, idx)       # [B, npoint, nsample, 1]
        return new_xyz, new_points, grouped_xyz_norm, idx, gd
    
class Norm(nj.Module):

  axis: tuple = (-1,)
  eps: float = 1e-4
  scale: bool = True
  shift: bool = True
  # for batchnorm
  momentum: float = 0.9
  use_running_stats: bool = True

  def __init__(self, impl):
    if '1em' in impl:
      impl, exp = impl.split('1em')
      self.eps = 10 ** -int(exp)
    self.impl = impl

  def __call__(self, x, train=True):
    ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    elif self.impl == 'batch':
      feat_axes = tuple(a % x.ndim for a in self.axis)
      reduce_axes = tuple(i for i in range(x.ndim) if i not in feat_axes)
      param_shape = [x.shape[i] if i in feat_axes else 1 for i in range(x.ndim)]

      if self.use_running_stats:
        if train:
          # Compute statistics over batch dimension
          batch_mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
          batch_mean = adc.checkpoint_name(batch_mean, 'small')
          
          # Compute variance
          batch_var = jnp.mean(jnp.square(x - batch_mean), axis=reduce_axes, keepdims=True)
          batch_var = adc.checkpoint_name(batch_var, 'small')
          
          # Update running statistics if needed
          if self.use_running_stats:
            # Update running mean and variance
            running_mean = self.get('running_mean', jnp.zeros, param_shape, f32)
            running_var = self.get('running_var', jnp.ones, param_shape, f32)
            new_running_mean = (1 - self.momentum) * running_mean + self.momentum * batch_mean
            new_running_var = (1 - self.momentum) * running_var + self.momentum * batch_var
            
            # Write new values to store
            self.write('running_mean', new_running_mean)
            self.write('running_var', new_running_var)
          
            # Use batch statistics for normalization during training
          mean, var = batch_mean, batch_var
        else:
          # Use running statistics during inference
          if self.use_running_stats:
            mean = self.get('running_mean', jnp.zeros, param_shape, f32)
            var = self.get('running_var', jnp.ones, param_shape, f32)
          else:
            # If not using running stats, compute batch statistics even in eval mode
            mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=reduce_axes, keepdims=True)

        # Get scale and shift parameters
        scale = self._scale(param_shape, x.dtype)
        shift = self._shift(param_shape, x.dtype)

        # Normalize
        x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)
    return self.get('scale', jnp.ones, shape, f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)
    return self.get('shift', jnp.zeros, shape, f32).astype(dtype)

class LayerNorm(Norm):
  def __init__(self, num_features, eps=1e-4, momentum=0.9, scale=True, shift=True, axis=(-1,)):
    super().__init__(impl='layer')
    self.axis = axis
    self.eps = eps
    self.momentum = momentum
    self.scale = scale
    self.shift = shift
    self.num_features = num_features
  
  def __call__(self, x, train=True):
    return super().__call__(x)

class RMSNorm(Norm):
  def __init__(self, num_features, eps=1e-4, momentum=0.9, scale=True, shift=True, axis=(-1,)):
    super().__init__(impl='rms')
    self.axis = axis
    self.eps = eps
    self.momentum = momentum
    self.scale = scale
    self.shift = shift
    self.num_features = num_features
  
  def __call__(self, x, train=True):
    return super().__call__(x)

class BatchNorm1d(Norm):
  """
  Input: (batch_size, num_features) or (batch_size, num_features, seq_length)
  """
  
  def __init__(self, num_features, eps=1e-4, momentum=0.9, scale=True, shift=True, use_running_stats=True):
    super().__init__(impl='batch')
    self.axis = (-1,)
    self.eps = eps
    self.momentum = momentum
    self.scale = scale
    self.shift = shift
    self.num_features = num_features
    self.use_running_stats = use_running_stats
        
  def __call__(self, x, train=True):
    assert x.ndim in (2, 3), f"Expected 2D or 3D input, got shape {x.shape}"
    if x.ndim == 2:
      assert x.shape[-1] == self.num_features, \
        f"Expected {self.num_features} features, got {x.shape[-1]}"
    else:  # x.ndim == 3
      assert x.shape[-1] == self.num_features, \
        f"Expected {self.num_features} features, got {x.shape[-1]}"
    
    return super().__call__(x, train=train)


class BatchNorm2d(Norm):
  """
  Input : (batch_size, height, width, num_channels)
  """
  def __init__(self, num_channels, eps=1e-4, momentum=0.9, scale=True, shift=True, use_running_stats=True):
    super().__init__(impl='batch')
    self.axis = (-1,)
    self.eps = eps
    self.momentum = momentum
    self.scale = scale
    self.shift = shift
    self.num_channels = num_channels
    self.use_running_stats = use_running_stats
      
  def __call__(self, x, train=True):
    assert x.ndim == 4, f"Expected 4D input, got shape {x.shape}"
    assert x.shape[-1] == self.num_channelsZz, \
        f"Expected {self.num_channelsZz} channels, got {x.shape[-1]}"
    
    return super().__call__(x, train=train)
  
class Conv1D(nj.Module):
  """
  1D Convolution module based on the Conv2D implementation.
  
  Args:
    depth: Number of output channels
    kernel: Kernel size (int)
    stride: Stride length (default 1)
  """
  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('normal') # is trunc_normal
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = kernel
    self.stride = stride

  def __call__(self, x):
    ensure_dtypes(x)
    # Kernel shape: [kernel_size, in_channels, out_channels]
    shape = (self.kernel, x.shape[-1] // self.groups, self.depth)
    kernel = self.get('kernel', self._scaled_winit, shape).astype(x.dtype)
    
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation for transposed convolution
      x = x.repeat(self.stride, -2)
      mask = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)[:, None]
      x *= mask
      stride = (1,)
    else:
      stride = (self.stride,)
    
    # Apply 1D convolution using lax.conv_general_dilated
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NHC', 'HIO', 'NHC'))  # 1D convolution dimension numbers
    
    if self.bias:
      x += self.get('bias', init(self.binit), self.depth).astype(x.dtype)
    
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale

import jax
import jax.numpy as jnp
from .networks import PointConv, PointConv_PCWM, Bottlenecked_PointConv_PCWM
import math
import einops
import numpy as np

import dreamerv3.ninjax as nj
from dreamerv3.jaxutils import symlog, DictConcat, cast_to_compute
from dreamerv3.nets import Linear, Conv2D, get_act
from pointclouds.nn_utils import Conv1D, Norm, RMSNorm, LayerNorm, BatchNorm1d
from pointclouds.networks import PointConv, PointConv_PCWM, Bottlenecked_PointConv_PCWM

class BasePointCloudEncoder(nj.Module):
    """Base class for point cloud encoders with shared functionality"""
    
    # Common parameters
    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False
    depth_max: int = 1000  # for RGBD
    
    def __init__(self, obs_space, **kw):
        assert all(len(s) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        assert 'pointcloud' in self.obs_space, "No key named 'pointcloud' in obs_space. Use simple encoder instead."
        excluded = ['reward', 'is_first', 'is_last', 'is_terminal', 'raw_pointcloud']
        obs_space = {k: s for k, s in obs_space.items() if k not in excluded}
        self.veckeys = [k for k, s in obs_space.items() if len(s) in (1,2) and k != 'pointcloud']
        self.imgkeys = [k for k, s in obs_space.items() if len(s) == 3]
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kw = kw
        self.shape = obs_space['pointcloud']
        
        # Subclass-specific initialization
        self._init_subclass_params()

    def _init_subclass_params(self):
        """Override in subclasses for specific initialization"""
        pass

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def _process_vector_inputs(self, obs, bdims):
        """Process vector inputs (common across all encoders)"""
        if not self.veckeys:
            return None
            
        vspace = {k: self.obs_space[k] for k in self.veckeys}
        vecs = {k: obs[k] for k in self.veckeys}
        squish = symlog if self.symlog else lambda x: x
        x = DictConcat(vspace, 1, squish=squish)(vecs)
        x = x.reshape((-1, *x.shape[bdims:]))
        
        for i in range(self.layers):
            x = self.get(f'mlp{i}', Linear, self.units, **self.kw)(x)
            x = get_act(self.act)(self.get(f'mlp{i}norm', Norm, self.norm)(x))
        return x

    def _process_image_inputs(self, obs, bdims):
        """Process image inputs (common across all encoders)"""
        if not self.imgkeys:
            return None
            
        K = self.kernel
        imgs = [obs[k] for k in sorted(self.imgkeys)]
        assert all(x.dtype == jnp.uint8 or x.shape[-1] == 4 for x in imgs)
        
        normed = []
        for img in imgs:
            if img.shape[-1] == 4:  # RGBD
                rgb, depth = img[..., :3], img[..., -1:]
                rgb = rgb.astype(jnp.float32) / 255.0
                depth = depth.astype(jnp.float32) / self.depth_max
                img = jnp.concatenate([rgb, depth], axis=-1)
            else:
                img = img.astype(jnp.float32) / 255.0
            normed.append(img)

        x = cast_to_compute(jnp.concatenate(normed, -1), force=True) - 0.5  # → (-0.5, +0.5)
        x = x.reshape((-1, *x.shape[bdims:]))
        
        for i, depth in enumerate(self.depths):
            if self.outer and i == 0:
                x = self.get(f'cnn{i}', Conv2D, depth, K, **self.kw)(x)
            elif self.strided:
                x = self.get(f'cnn{i}', Conv2D, depth, K, 2, **self.kw)(x)
            else:
                x = self.get(f'cnn{i}', Conv2D, depth, K, **self.kw)(x)
                B, H, W, C = x.shape
                x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
            x = get_act(self.act)(self.get(f'cnn{i}norm', Norm, self.norm)(x))
        
        assert 3 <= x.shape[-3] <= 16, x.shape
        assert 3 <= x.shape[-2] <= 16, x.shape
        x = x.reshape((x.shape[0], -1))
        return x

    def _process_pointcloud(self, obs, bdims, training):
        """Process point cloud data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _process_pointcloud")

    def __call__(self, obs, training):
        # bdims = 1 if single else 2
        # bshape = reset.shape
        bshape = obs['pointcloud'].shape[:-len(self.shape)]
        bdims = 2 if len(bshape)==2 else 1
        outs = []

        # Process vector inputs
        vec_features = self._process_vector_inputs(obs, bdims)
        if vec_features is not None:
            outs.append(vec_features)

        # Process image inputs
        img_features = self._process_image_inputs(obs, bdims)
        if img_features is not None:
            outs.append(img_features)

        # Process point cloud
        pcd_features = self._process_pointcloud(obs, bdims, training)
        outs.append(pcd_features)

        # Combine all features
        x = jnp.concatenate(outs, -1)
        tokens = x.reshape((*bshape, *x.shape[1:]))

        return tokens


class PointConvEncoder(BasePointCloudEncoder):
    """Original PointConv implementation"""
    
    # PointConv specific parameters
    bandwidths: tuple = (0.05, 0.1, 0.2, 0.4)
    npoints: tuple = (512, 256, 128, 1)
    nsamples: tuple = (16, 32, 64, None)
    mlps: tuple = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 256))
    pool: str = 'mean'
    pcd_norm: str = 'layer'

    def _process_pointcloud(self, obs, bdims, training):
        points = obs['pointcloud']
        points = cast_to_compute(points)
        
        if len(points.shape) == bdims + 2:  # [B, N, C] or [B, T, N, C]
            xyz = points[..., :3]
            features = points[..., 3:] / 255. if points.shape[-1] > 3 else None
        else:
            raise ValueError(f"Unexpected point cloud shape: {points.shape}")
        
        xyz = xyz.reshape(-1, *xyz.shape[bdims:])
        if features is not None:
            features = features.reshape(-1, *features.shape[bdims:])

        for i in range(len(self.npoints)):
            xyz, features = self.get(
                f'pointconv{i}',
                PointConv,
                npoint=self.npoints[i],
                nsample=self.nsamples[i],
                in_channel=xyz.shape[1] + (0 if features is None else features.shape[1]),
                mlp=self.mlps[i],
                bandwidth=self.bandwidths[i],
                group_all=self.nsamples[i] == None,
                norm=self.pcd_norm,
            )(xyz, features, training)

        return jnp.squeeze(features, 1)


class PointConvEncoder_PCWM(BasePointCloudEncoder):
    """Enhanced PointConv implementation with presampling"""
    
    # Enhanced PointConv parameters
    in_points: int = 1024
    npoints: tuple = (512, 256, 128, 64)
    nsamples: tuple = (16, 16, 16, 16)
    mlps: tuple = (16, 16)
    out_channels: tuple = (32, 64, 128, 256)
    pool: str = 'mean'
    pcd_norm: str = 'layer'
    pcd_act: str = 'leakyrelu'
    presample: bool = True

    def _init_subclass_params(self):
        if self.presample:
            n_points = [self.in_points] + list(self.npoints)
            cum = np.cumsum(n_points).tolist()
            self._split_idxs = cum[:-1]

    def _process_pointcloud(self, obs, bdims, training):
        points = obs['pointcloud']
        
        if self.presample:
            assert points.shape[-2] == self.in_points + sum(self.npoints)
        
        points = cast_to_compute(points)
        xyz = points[..., :3]
        xyz = xyz.reshape(-1, *xyz.shape[bdims:])

        init_features = points[..., :self.in_points, 3:] / 255. if points.shape[-1] > 3 else None
        if init_features is not None:
            init_features = init_features.reshape(-1, *init_features.shape[bdims:])
            init_features = jnp.concatenate([xyz[:, :self.in_points], init_features], axis=-1)
        else:
            init_features = xyz[:, :self.in_points]

        features = init_features
        for i, layer in enumerate(self.mlps):
            features = self.get(f'mlp{i}', Linear, layer)(features)
            features = jax.nn.leaky_relu(self.get(f'mlp{i}norm', Norm, 'layer')(features), 0.2)

        if self.presample:
            xyz_segments = jnp.split(xyz, self._split_idxs, axis=1)

        for i in range(len(self.npoints)):
            inputs = xyz_segments[i] if self.presample else xyz
            new_inputs = xyz_segments[i+1] if self.presample else None
            
            if i == 0:
                xyz, features = self.get(
                    f'pointconv{i}',
                    PointConv_PCWM,
                    npoint=self.npoints[i],
                    nsample=self.nsamples[i],
                    in_channel=features.shape[-1],
                    out_channel=self.out_channels[i],
                    norm=self.pcd_norm,
                    act=self.pcd_act,
                )(inputs, features, new_inputs, None, training)
            else:
                xyz, features = self.get(
                    f'pointconv{i}',
                    Bottlenecked_PointConv_PCWM,
                    npoint=self.npoints[i],
                    nsample=self.nsamples[i],
                    in_channel=features.shape[-1],
                    out_channel=self.out_channels[i],
                    bottleneck=4,
                    norm=self.pcd_norm,
                    act=self.pcd_act,
                )(inputs, features, new_inputs, None, training)

        if self.pool == 'mean':
            return jnp.mean(features, axis=-2)
        else:
            raise NotImplementedError


class PointNetEncoder(BasePointCloudEncoder):
    """PointNet implementation following https://arxiv.org/pdf/1612.00593"""
    
    # PointNet specific parameters
    mlps: tuple = (128, 128, 256)
    pcd_norm: str = 'layer'
    pcd_act: str = 'relu'
    out_dim: int = 128
    ignore_first_ln: bool = True

    def _init_subclass_params(self):
        if self.pcd_norm == 'batch':
            self.pcd_norm_fn = BatchNorm1d
        elif self.pcd_norm == 'layer':
            self.pcd_norm_fn = LayerNorm
        elif self.pcd_norm == 'rms':
            self.pcd_norm_fn = RMSNorm
        else:
            raise NotImplementedError

    def _process_pointcloud(self, obs, bdims, training):
        points = obs['pointcloud']

        if points.shape[-1] > 3:
            rgb = points[..., 3:] / 255.  # rgb feature
            points = jnp.concatenate([points[..., :3], rgb], axis=-1)

        points = cast_to_compute(points)
        points = points.reshape(-1, *points.shape[bdims:])
        features = points

        for i, out_dim in enumerate(self.mlps):
            features = self.get(f'conv1d{i}', Conv1D, out_dim, 1)(features)
            if i != 0 or not self.ignore_first_ln:
                features = self.get(f'norm{i}', self.pcd_norm_fn, out_dim)(features)
            features = get_act(self.pcd_act)(features)

        global_feat = jnp.max(features, axis=-2)
        return self.get(f'linear', Linear, self.out_dim)(global_feat)
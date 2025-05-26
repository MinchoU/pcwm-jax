import jax.numpy as jnp

import jax
from .nn_utils import compute_density, sample_and_group_all, _sample_and_group_jit, knn_point, index_points, index_points_3d, _fps_jit

jdb = jax.debug.breakpoint
sg = jax.lax.stop_gradient

from pointclouds.nn_utils import BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm, Conv1D
from dreamerv3.nets import Conv2D, Linear
import dreamerv3.ninjax as nj

class DensityNet(nj.Module):
    """
    Density network for density weighting
    """
    
    def __init__(self, hidden_unit=[16, 8], norm=LayerNorm, act=jax.nn.relu):
        super().__init__()
        self.hidden_unit = hidden_unit
        self.norm_fn = norm
        self.act = act
        # conv2d : B, H, W, Cin -> B, W, H, Cout
        # batchnorm2d : B, H, W, C

    def __call__(self, density_scale, train):
        """
        Input:
            density_scale: [B, N, nsample, 1]
        Output:
            density_scale: [B, N, nsample, 1]
        """
        x = density_scale

        for i in range(len(self.hidden_unit)):
            x = self.get(f'conv{i}', Conv2D, self.hidden_unit[i], 1)(x)
            x = self.get(f'norm{i}', self.norm_fn, self.hidden_unit[i])(x, train)
            x = self.act(x)
        
        x = self.get('conv_out', Conv2D, 1, 1)(x)
        x = self.get('norm_out', self.norm_fn, 1)(x, train)
        x = self.act(x)
        # x = jax.nn.sigmoid(x)

        return x

class WeightNet(nj.Module):
    """
    Weight network for learning point weights
    """
    
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], norm=LayerNorm, act=jax.nn.relu):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_unit = hidden_unit
        self.norm_fn = norm
        self.act = act
    
    def __call__(self, localized_xyz, train):
        """
        Input:
            localized_xyz: [B, C, nsample, N]
        Output:
            weights: [B, out_channel, nsample, N]
        """
        weights = localized_xyz
        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            weights = self.get('conv0', Conv2D, self.out_channel, 1)(weights)
            weights = self.get('norm0', self.norm_fn, self.out_channel)(weights, train)
            weights = self.act(weights)
        
        else:
            for i in range(len(self.hidden_unit)):
                weights = self.get(f'conv{i}', Conv2D, self.hidden_unit[i], 1)(weights)
                weights = self.get(f'norm{i}', self.norm_fn, self.hidden_unit[i])(weights, train)
                weights = self.act(weights)
            
            weights = self.get('conv_out', Conv2D, self.out_channel, 1)(weights)
            weights = self.get('norm_out', self.norm_fn, self.out_channel)(weights, train)
            weights = self.act(weights)

        return weights
    
class PointConv(nj.Module):
    """
    PointConv with density estimation
    """
    
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all, norm):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.bandwidth = bandwidth
        self.group_all = group_all
        self.in_channel = in_channel
        if norm == 'batch':
            self.norm_fn_2d = BatchNorm2d
            self.norm_fn_1d = BatchNorm1d
        elif norm == 'layer':
            self.norm_fn_1d, self.norm_fn_2d = LayerNorm, LayerNorm
        elif norm == 'rms':
            self.norm_fn_1d, self.norm_fn_2d = RMSNorm, RMSNorm
        else:
            raise NotImplementedError
        
        self.weight_net = WeightNet(3, 16, name='wnet', norm=self.norm_fn_2d)
        self.density_net = DensityNet(name='dnet', norm=self.norm_fn_2d)
    
    def __call__(self, xyz, points, train=True):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C_in]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, C_out]
        """        
        # Compute density
        B, N, C = xyz.shape
        # xyz_density = nj.checkpoint(compute_density)(xyz, self.bandwidth)
        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / (xyz_density)
        
        # Sample and group
        def group(xyz, points, inv_den):
            if self.group_all:
                new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(
                    xyz, points, jnp.expand_dims(inv_den, -1)
                )
            else:
                # new_xyz, new_points, grouped_xyz_norm, _, grouped_density = _sample_and_group_jit(
                #     self.npoint, self.nsample, xyz, points, jnp.expand_dims(inv_den, -1)
                # )
                new_xyz, new_points, grouped_xyz_norm, _, grouped_density = _sample_and_group_jit(
                    self.npoint, self.nsample, xyz, points, None, None, jnp.expand_dims(inv_den, -1)
                )
        
        
            return new_xyz, new_points, grouped_xyz_norm, grouped_density

        new_xyz, new_points, grouped_xyz_norm, grouped_density = group(xyz, points, inverse_density)
        # B*N_new (N)

        def mlp_block(x):
            for i, out_channel in enumerate(self.mlp):
                x = self.get(f'mlp_conv{i}', Conv2D, out_channel, 1)(x) # [B, npoint, nsample, outchannel]
                x = self.get(f'mlp_norm{i}', self.norm_fn_2d, out_channel)(x, train)
                x = jax.nn.relu(x)
            return x
        
        # new_points : B*N*nsample*(3+n_feat)
        x = mlp_block(new_points) 

        # Apply density weighting
        def density(x, grouped_density):
            inverse_max_density = jnp.max(grouped_density, axis=2, keepdims=True)
            density_scale = grouped_density / (inverse_max_density)
            
            # Apply density network
            density_scale = self.density_net(density_scale, train)
        
            # Apply weights
            x = x * density_scale
            return x
        
        x = density(x, grouped_density)
        
        # x : B*npoint*nsample*
        # Apply weight network
        def weight(x, grouped_xyz_norm):
            weights = self.weight_net(grouped_xyz_norm, train) # B*N*n_sample*C_mid
            # Matrix multiplication]
            # jax.debug.breakpoint()
            x = jnp.transpose(x, (0, 1, 3, 2))
            x = jnp.matmul(x,weights)
            
            return x
        
        x = weight(x, grouped_xyz_norm)
        
        # Reshape and apply linear layer (final 1*1 conv)
        x = x.reshape(B, self.npoint, -1)
        x = self.get('conv_out', Linear, self.mlp[-1])(x)

        x = self.get('norm_out', self.norm_fn_1d, self.mlp[-1])(x, train)
        x = jax.nn.relu(x)

        return new_xyz, x
    
class PointConv_PCWM(nj.Module):
    """
    PointConv with density estimation
    """
    def __init__(self, npoint, nsample, in_channel, out_channel, norm, act):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.in_channel = in_channel
        self.out_channel = out_channel

        if norm == 'batch':
            self.norm_fn_2d = BatchNorm2d
            self.norm_fn_1d = BatchNorm1d
        elif norm == 'layer':
            self.norm_fn_1d, self.norm_fn_2d = LayerNorm, LayerNorm
        elif norm == 'rms':
            self.norm_fn_1d, self.norm_fn_2d = RMSNorm, RMSNorm
        else:
            raise NotImplementedError
        
        if act == 'relu':
            self.act = jax.nn.relu
        elif act == 'gelu':
            self.act = jax.nn.gelu
        elif act == 'leakyrelu':
            self.act = lambda x: jax.nn.leaky_relu(x, negative_slope=0.2)
        else:
            raise NotImplementedError
        
        self.weight_net = WeightNet(3, 16, name='wnet', norm=self.norm_fn_2d, act=self.act)
    
    def __call__(self, xyz, points, new_xyz=None, nn_idx=None, train=True):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C_in]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, C_out]
        """        
        # Compute density
        B, N, C = xyz.shape
        
        # Sample and group
        def group(xyz, points, new_xyz, nn_idx):
            new_xyz, new_points, grouped_xyz_norm, _, _ = _sample_and_group_jit(
                self.npoint, self.nsample, xyz, points, new_xyz, nn_idx
            )
        
            return new_xyz, new_points, grouped_xyz_norm

        new_xyz, new_points, grouped_xyz_norm = group(xyz, points, new_xyz, nn_idx)

        x = new_points
        # x : B*npoint*nsample*
        # Apply weight network
        def weight(x, grouped_xyz_norm):
            weights = self.weight_net(grouped_xyz_norm, train) # B*N*n_sample*C_mid
            # Matrix multiplication]
            # jax.debug.breakpoint()
            x = jnp.transpose(x, (0, 1, 3, 2))
            x = jnp.matmul(x,weights)
            return x
        
        x = weight(x, grouped_xyz_norm)
        
        # Reshape and apply linear layer (final 1*1 conv)
        x = x.reshape(B, self.npoint, -1)
        x = self.get('conv_out', Linear, self.out_channel)(x)

        x = self.get('norm_out', self.norm_fn_1d, self.out_channel)(x, train)
        x = self.act(x)

        return new_xyz, x
    
class Bottlenecked_PointConv_PCWM(nj.Module):
    """
    PointConv with density estimation
    """
    def __init__(self, npoint, nsample, in_channel, out_channel, bottleneck, norm, act):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bottleneck = bottleneck
        self.norm = norm
        self.act = act

        if norm == 'batch':
            self.norm_fn_2d = BatchNorm2d
            self.norm_fn_1d = BatchNorm1d
        elif norm == 'layer':
            self.norm_fn_1d, self.norm_fn_2d = LayerNorm, LayerNorm
        elif norm == 'rms':
            self.norm_fn_1d, self.norm_fn_2d = RMSNorm, RMSNorm
        else:
            raise NotImplementedError
        
        if act == 'relu':
            self.act_fn = jax.nn.relu
        elif act == 'gelu':
            self.act_fn = jax.nn.gelu
        elif act == 'leakyrelu':
            self.act_fn = lambda x: jax.nn.leaky_relu(x, negative_slope=0.2)
        else:
            raise NotImplementedError
        
        self.weight_net = WeightNet(3, 16, name='wnet', norm=self.norm_fn_2d, act=self.act_fn)
        if in_channel == out_channel:
            self.residual = lambda x:x
        else:
            self.residual = self.get('residual', Linear, out_channel)

    
    def __call__(self, xyz, points, new_xyz, nn_idx=None, train=True):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C_in]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, C_out]
        """        
        # Compute density
        B, N, C = xyz.shape
        if new_xyz is None:
            fps_idx, _ = _fps_jit(xyz, self.npoint)             # [B, npoint]
            new_xyz = sg(index_points(xyz, fps_idx))           # [B, npoint, C]
            # new_xyz = index_points(xyz, fps_idx)           # [B, npoint, C]

        if nn_idx is None:
            nn_idx = knn_point(self.nsample, xyz, new_xyz)
        
        reduced = self.get('reduce_lin', Linear, self.in_channel//self.bottleneck)(points)
        reduced = self.get('reduce_norm', self.norm_fn_1d, self.in_channel//self.bottleneck)(reduced, train)
        reduced = self.act_fn(reduced)

        new_xyz, new_points = self.get('inner_pointconv', 
                              PointConv_PCWM,
                              self.npoint,
                              self.nsample,
                              self.in_channel//self.bottleneck,
                              self.out_channel//self.bottleneck,
                              norm=self.norm,
                              act=self.act)(xyz, reduced, new_xyz, nn_idx, train)
        
        new_points = self.get('expand_lin', Linear, self.out_channel)(new_points)
        new_points = self.get('expand_norm', self.norm_fn_1d, self.out_channel)(new_points, train)

        group_points = index_points_3d(points, nn_idx) # B*npoint*nsample*C
        points = jnp.mean(group_points, axis=2)
        shortcut = self.residual(points)
        new_points = self.act_fn(new_points+shortcut)

        return new_xyz, new_points
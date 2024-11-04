import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        assert len(size) == 2 or len(size) == 3, "SpatialTransformer accepts 2D or 3D inputs."

        self.mode = mode

        if len(size) == 2:
            h, w = size

            x_axis = torch.arange(w, dtype=torch.int32)
            y_axis = torch.arange(h, dtype=torch.int32)
            grid = torch.meshgrid([x_axis, y_axis], indexing="xy")
            grid = torch.stack(grid)
            grid = torch.unsqueeze(grid, 0).float()

            xscale = torch.ones((h, w), dtype=torch.float32)
            xscale.fill_((w - 1) / 2.0)
            yscale = torch.ones((h, w), dtype=torch.float32)
            yscale.fill_((h - 1) / 2.0)
            scale = torch.stack([xscale, yscale], dim=0)
        elif len(size) == 3:
            d, h, w = size

            x_axis = torch.arange(w, dtype=torch.int32)
            y_axis = torch.arange(h, dtype=torch.int32)
            z_axis = torch.arange(d, dtype=torch.int32)
            grid = torch.meshgrid([z_axis, y_axis, x_axis], indexing="ij")
            grid = torch.stack([grid[2], grid[1], grid[0]])
            grid = torch.unsqueeze(grid, 0).float()

            xscale = torch.ones((d, h, w), dtype=torch.float32)
            xscale.fill_((w - 1) / 2.0)
            yscale = torch.ones((d, h, w), dtype=torch.float32)
            yscale.fill_((h - 1) / 2.0)
            zscale = torch.ones((d, h, w), dtype=torch.float32)
            zscale.fill_((d - 1) / 2.0)
            scale = torch.stack([xscale, yscale, zscale], dim=0)
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, src: torch.Tensor, displacement_flow: torch.Tensor):
        # new locations
        new_locs: torch.Tensor = self.grid + displacement_flow

        # need to normalize grid values to [-1, 1] for resampler
        new_locs = new_locs / self.scale - 1.0

        # (bs, 2, h, w) or (bs, 3, d, h, w)
        shape = displacement_flow.shape[2:]

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            # 需要调整flow, 适合grid sample [bs, h, w, [x,y]]
            new_locs = new_locs.permute(0, 2, 3, 1)
            # new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            #  需要调整flow, 适合grid sample [bs, d, h, w, [x,y,z]]
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            # new_locs = new_locs[..., [2, 1, 0]]

        bs = src.shape[0]
        if bs > 1:
            size = list(new_locs.size())
            size[0] = bs
            new_locs = new_locs.expand(size, implicit=True)

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = "linear"
        if ndims == 2:
            self.mode = "bi" + self.mode
        elif ndims == 3:
            self.mode = "tri" + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

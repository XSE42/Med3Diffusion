import torch
import torch.nn as nn

class HessianReg2D(nn.Module):
    """
    Regularization term of bending energy of a 2D displacement field

    Parameters:
        spacing: spacing in (H, W) order rather than (x, y) order
        shape: shape is (H, W)
    """

    def __init__(self, norm="L2", spacing=(1, 1), shape=(176, 144), normalize=True, device=torch.device("cpu")):
        super().__init__()
        self.norm = norm
        self.spacing = torch.tensor(spacing[::-1]).float().to(device)  # self.spacing is in (x, y) order

        h, w = shape
        xscale = torch.ones((h, w), dtype=torch.float32)
        xscale.fill_((w - 1) / 2.0)
        yscale = torch.ones((h, w), dtype=torch.float32)
        yscale.fill_((h - 1) / 2.0)
        self.scale = torch.stack([xscale, yscale], dim=0).to(device)

        self.spatial_dims = torch.tensor(shape[::-1]).float().to(device)  # self.spatial_dims is in (x, y) order
        self.normalize = normalize
        if self.normalize:
            self.spacing /= self.spacing.min()
            self.spatial_dims /= self.spatial_dims.min()

    def forward(self, flow):
        """
        Parameters:
            flow: [N, (x, y), H, W], notice that dim=1 is in (x, y) order rather than (y, x) order
                In original repo uncbiag/Aladdin, the order of flow is [N, (y, x), H, W] or [N, (z, y, x), D, H, W],
                and the SpatialTransformer in that repo will flip the order of dim=1 from (y, x) or (z, y, x) to (x, y)
                or (x, y, z) before passing to the F.grid_sample function.
                In this repo, our SpatialTransformer will not flip the order of dim=1, so our flow is [N, (x, y), H, W],
                and we need to flip the order of spacing and shape in the __init__ function.
        """

        bs = flow.shape[0]
        channel = flow.shape[1]

        flow = flow / self.scale - 1.0

        # f''(y) = [f(y+h) + f(y-h) - 2f(y)] / h^2
        ddy = torch.abs(flow[:, :, 2:, 1:-1] + flow[:, :, :-2, 1:-1] - 2 * flow[:, :, 1:-1, 1:-1]).view(bs, channel, -1)

        # f''(x) = [f(x+h) + f(x-h) - 2f(x)] / h^2
        ddx = torch.abs(flow[:, :, 1:-1, 2:] + flow[:, :, 1:-1, :-2] - 2 * flow[:, :, 1:-1, 1:-1]).view(bs, channel, -1)

        # f_{y, x}(y, x) = [df(y+h, x+k) + df(y-h, x-k) - df(y+h, x-k) - df(y-h, x+k)] / 2hk
        dydx = torch.abs(flow[:, :, 2:, 2:] + flow[:, :, :-2, :-2] -
                         flow[:, :, 2:, :-2] - flow[:, :, :-2, 2:]).view(bs, channel, -1)

        if self.norm == "L2":
            ddy = (ddy ** 2).mean(2) * (self.spatial_dims * self.spacing / (self.spacing[1] ** 2)) ** 2
            ddx = (ddx ** 2).mean(2) * (self.spatial_dims * self.spacing / (self.spacing[0] ** 2)) ** 2
            dydx = (dydx ** 2).mean(2) * (self.spatial_dims * self.spacing / (self.spacing[0] * self.spacing[1])) ** 2

        d = (ddx.mean() + ddy.mean() + 2 * dydx.mean()) / 4.0
        return d


class JacobianDeterminant(nn.Module):
    """
    Jacobian determinant of a 2D deformation displacement field
    Do NOT add identity field to the displacement field before passing it to this module
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow):
        """
        Parameters:
            flow: [N, (x, y), H, W], displacement field
        """

        bs = flow.shape[0]

        # f'(y) = [f(y+h) - f(y-h)] / 2h
        dy = (flow[:, :, 2:, 1:-1] - flow[:, :, :-2, 1:-1]) / 2.0

        # f'(x) = [f(x+h) - f(x-h)] / 2h
        dx = (flow[:, :, 1:-1, 2:] - flow[:, :, 1:-1, :-2]) / 2.0

        fxdx = dx[:, 0, :, :] + 1.0
        fxdy = dy[:, 0, :, :]
        fydx = dx[:, 1, :, :]
        fydy = dy[:, 1, :, :] + 1.0

        det = fxdx * fydy - fxdy * fydx
        det = det.view(bs, -1)
        mask = det < 0
        num_folding = mask.sum(dim=1, dtype=torch.float32)

        return num_folding


class JacobianDeterminant3D(nn.Module):
    """
    Jacobian determinant of a 3D deformation displacement field
    Do NOT add identity field to the displacement field before passing it to this module
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow):
        """
        Parameters:
            flow: [N, (x, y, z), D, H, W], displacement field
        """

        bs = flow.shape[0]

        # f'(z) = [f(z+h) - f(z-h)] / 2h
        dz = (flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]) / 2.0

        # f'(y) = [f(y+h) - f(y-h)] / 2h
        dy = (flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]) / 2.0

        # f'(x) = [f(x+h) - f(x-h)] / 2h
        dx = (flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]) / 2.0

        fxdx = dx[:, 0] + 1.0
        fxdy = dy[:, 0]
        fxdz = dz[:, 0]
        fydx = dx[:, 1]
        fydy = dy[:, 1] + 1.0
        fydz = dz[:, 1]
        fzdx = dx[:, 2]
        fzdy = dy[:, 2]
        fzdz = dz[:, 2] + 1.0

        sub_det1 = fydy * fzdz - fydz * fzdy
        sub_det2 = fydx * fzdz - fydz * fzdx
        sub_det3 = fydx * fzdy - fydy * fzdx
        det = fxdx * sub_det1 - fxdy * sub_det2 + fxdz * sub_det3
        det = det.view(bs, -1)
        mask = det < 0
        num_folding = mask.sum(dim=1, dtype=torch.float32)

        return num_folding

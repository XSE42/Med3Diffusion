from typing import Tuple, List

import scipy
import torch
import torch.nn.functional as F
import numpy as np
import monai.metrics as metrics


SYNTHSEG_LABELS = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
]
NONE_SYNTHSEG_LABELS = [i for i in range(61) if i not in SYNTHSEG_LABELS]
SPACING = (1.0, 1.0, 1.0)


def get_onehot_mask(
        seg_mask: torch.Tensor,
        num_classes: int = 61,
        valid_labels: List[int] = SYNTHSEG_LABELS,
    ) -> torch.LongTensor:
    """
    Parameters
        seg_mask (`torch.Tensor` of shape `(BS, H, W)` or `(BS, D, H, W)`): segmentation mask
        num_classes (`int`): number of classes
    Return
        seg_mask_onehot (`torch.LongTensor` of shape `(BS, C, H, W)` or `(BS, C, D, H, W)`):
            onehot segmentation mask, C is the number of valid classes
    """
    seg_mask_onehot = F.one_hot(seg_mask.long(), num_classes=num_classes)
    if len(seg_mask_onehot.shape) == 4:
        seg_mask_onehot = seg_mask_onehot.permute(0, 3, 1, 2)  # (BS, C_in, H, W)
    else:
        seg_mask_onehot = seg_mask_onehot.permute(0, 4, 1, 2, 3)  # (BS, C_in, D, H, W)
    seg_mask_onehot = seg_mask_onehot[:, valid_labels]  # (BS, C, D, H, W), get valid classes
    return seg_mask_onehot


@torch.no_grad()
def metrcis_onehot(
        pred_onehot: torch.FloatTensor,
        gt_onehot: torch.LongTensor,
        spacing: Tuple[int] = SPACING,
        device: torch.device = torch.device("cpu"),
    ):
    """
    Parameters
        pred_onehot (`torch.LongTensor` of shape `(BS, C, H, W)` or `(BS, C, D, H, W)`):
            predicted segmentation mask
        gt_onehot (`torch.LongTensor` of shape `(BS, C, H, W)` or `(BS, C, D, H, W)`):
            ground truth segmentation mask
        spacing (`Tuple[int]`): pixel spacing
        device (`torch.device`): result device
    """
    num_classes = pred_onehot.shape[1]
    results = {}

    # Dice Score
    dice = metrics.compute_dice(
        pred_onehot,
        gt_onehot,
        include_background=False,
        num_classes=num_classes,
    ).to(device)  # (BS, C)
    results["dice"] = dice

    # Average Surface Distance
    asd = metrics.compute_average_surface_distance(
        pred_onehot,
        gt_onehot,
        symmetric=True,
        spacing=spacing,
    ).to(device)  # (BS, C)
    results["asd"] = asd

    # Hausdorff Distance
    hd = metrics.compute_hausdorff_distance(
        pred_onehot,
        gt_onehot,
        spacing=spacing,
    ).to(device)  # (BS, C)
    results["hd"] = hd

    # IoU
    iou = metrics.compute_iou(
        pred_onehot,
        gt_onehot,
        include_background=False,
    ).to(device)  # (BS, C)
    results["iou"] = iou

    return results


# monai.metrics.fid.get_fid_score
def get_fid_score(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the FID score metric on a batch of feature vectors.

    Args:
        y_pred: feature vectors extracted from a pretrained network run on generated images.
        y: feature vectors extracted from a pretrained network run on images from the real data distribution.
    """
    y = y.double()
    y_pred = y_pred.double()

    if y.ndimension() > 2:
        raise ValueError("Inputs should have (number images, number of features) shape.")

    mu_y_pred = torch.mean(y_pred, dim=0)
    sigma_y_pred = _cov(y_pred, rowvar=False)
    mu_y = torch.mean(y, dim=0)
    sigma_y = _cov(y, rowvar=False)

    return compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y, sigma_y)


def _cov(input_data: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    """
    Estimate a covariance matrix of the variables.

    Args:
        input_data: A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable,
            and each column a single observation of all those variables.
        rowvar: If rowvar is True (default), then each row represents a variable, with observations in the columns.
            Otherwise, the relationship is transposed: each column represents a variable, while the rows contain
            observations.
    """
    if input_data.dim() < 2:
        input_data = input_data.view(1, -1)

    if not rowvar and input_data.size(0) != 1:
        input_data = input_data.t()

    factor = 1.0 / (input_data.size(1) - 1)
    input_data = input_data - torch.mean(input_data, dim=1, keepdim=True)
    return factor * input_data.matmul(input_data.t()).squeeze()


def _sqrtm(input_data: torch.Tensor) -> torch.Tensor:
    """Compute the square root of a matrix."""
    scipy_res, _ = scipy.linalg.sqrtm(input_data.detach().cpu().numpy().astype(np.float_), disp=False)
    return torch.from_numpy(scipy_res)


def compute_frechet_distance(
    mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_y: torch.Tensor, sigma_y: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """The Frechet distance between multivariate normal distributions."""
    diff = mu_x - mu_y

    covmean = _sqrtm(sigma_x.mm(sigma_y))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f"FID calculation produces singular product; adding {epsilon} to diagonal of covariance estimates")
        offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * epsilon
        covmean = _sqrtm((sigma_x + offset).mm(sigma_y + offset))

    # Numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        # if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0, dtype=torch.double), atol=1e-3):
        #     raise ValueError(f"Imaginary component {torch.max(torch.abs(covmean.imag))} too high.")
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean

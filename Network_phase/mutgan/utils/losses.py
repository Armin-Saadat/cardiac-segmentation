import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.one_hot import one_hot


class Dice(nn.Module):
    """
    Dice-Loss for segmentation.
    Shape of data: (bs, H, W)
    """

    def loss(self, target, pred):
        vol_axes = [1, 2]
        top = 2 * (target * pred).sum(dim=vol_axes)
        bottom = torch.clamp((target + pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    Shape of data: (bs, H, W)
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, target, pred):
        Ii = target
        Ji = pred
        ndims = 2

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")
        pad_no = math.floor(win[0] / 2)
        padding = (pad_no, pad_no)
        stride = (1, 1)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE(nn.Module):
    """
    Mean squared error loss.
    Shape of data: (bs, H, W)
    """

    def loss(self, target, pred):
        return torch.mean((target - pred) ** 2)


class Accuracy(nn.Module):
    """
    Acc-Loss for segmentation
    Shape of data: (bs, H, W)
    """

    def loss(self, target, pred):
        acc = torch.sum(pred == target)
        return -(acc / len(pred.view(-1)))


class FocalLoss(nn.Module):
    r"""According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
         N = 5  # num_classes
         kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
         criterion = FocalLoss(**kwargs)
         input = torch.randn(1, N, 3, 5, requires_grad=True)
         target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
         output = criterion(input, target)
         output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

    def focal_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            alpha: float,
            gamma: float = 2.0,
            reduction: str = 'none',
            eps: float = 1e-8,
    ) -> torch.Tensor:
        r"""Criterion that computes Focal loss.
        According to :cite:`lin2018focal`, the Focal loss is computed as follows:
        .. math::
            \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
        Where:
           - :math:`p_t` is the model's estimated probability for each class.
        Args:
            input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
            target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
            alpha: Weighting factor :math:`\alpha \in [0, 1]`.
            gamma: Focusing parameter :math:`\gamma >= 0`.
            reduction: Specifies the reduction to apply to the
              output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
              will be applied, ``'mean'``: the sum of the output will be divided by
              the number of elements in the output, ``'sum'``: the output will be
              summed.
            eps: Scalar to enforce numerical stabiliy.
        Return:
            the computed loss.
        Example:
             N = 5  # num_classes
             input = torch.randn(1, N, 3, 5, requires_grad=True)
             target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
             output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
             output.backward()
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) >= 2:
            raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

        if input.size(0) != target.size(0):
            raise ValueError(
                f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

        n = input.size(0)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError(f'Expected target size {out_size}, got {target.size()}')

        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

        # compute softmax over the classes axis
        input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device,
                                               dtype=input.dtype)

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, gamma)

        focal = -alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
    Shape:
        - Input: :math:`(N, 1, *)`.
        - Target: :math:`(N, 1, *)`.
    Examples:
         N = 1  # num_classes
         kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
         loss = BinaryFocalLossWithLogits(**kwargs)
         input = torch.randn(1, N, 3, 5, requires_grad=True)
         target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
         output = loss(input, target)
         output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, self.reduction, self.eps)

    def binary_focal_loss_with_logits(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = 'none',
            eps: float = 1e-8,
    ) -> torch.Tensor:
        r"""Function that computes Binary Focal loss.
        .. math::
            \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
        where:
           - :math:`p_t` is the model's estimated probability for each class.
        Args:
            input: input data tensor with shape :math:`(N, 1, *)`.
            target: the target tensor with shape :math:`(N, 1, *)`.
            alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
            gamma: Focusing parameter :math:`\gamma >= 0`.
            reduction: Specifies the reduction to apply to the
              output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
              will be applied, ``'mean'``: the sum of the output will be divided by
              the number of elements in the output, ``'sum'``: the output will be
              summed.
            eps: for numerically stability when dividing.
        Returns:
            the computed loss.
        Examples:
             num_classes = 1
             kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
             logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
             labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
             binary_focal_loss_with_logits(logits, labels, **kwargs)
            tensor(4.6052)
        """

        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) >= 2:
            raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

        if input.size(0) != target.size(0):
            raise ValueError(
                f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

        probs = torch.sigmoid(input)
        target = target.unsqueeze(dim=1)
        loss_tmp = -alpha * torch.pow((1.0 - probs + eps), gamma) * target * torch.log(probs + eps) - (
                1 - alpha
        ) * torch.pow(probs + eps, gamma) * (1.0 - target) * torch.log(1.0 - probs + eps)

        loss_tmp = loss_tmp.squeeze(dim=1)

        if reduction == 'none':
            loss = loss_tmp
        elif reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {reduction}")
        return loss

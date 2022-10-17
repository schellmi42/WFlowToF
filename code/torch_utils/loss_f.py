'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
import torch.nn.functional as F
from torch_utils.warp import twotap_reordering


def latent_similarity(f1, f2, type='cost'):
  """ computes a similarity loss of two vectors
  Args:
    f1: [B, C, H, W]
    f2: [B, C, H, W]
  Returns:
    float
  """
  # C = f1.shape[1]
  if type == 'cost':
    return (f1 * f2).mean()
  elif type == 'cosine' or 'cos':
    return (F.normalize(f1, dim=1) * F.normalize(f2, dim=1)).abs().mean()
  elif type == 'L1' or type == 'l1' or type == 'MAE':
    return (f1 - f2).abs().mean()
  elif type == 'L2' or type == 'l2' or type == 'MSE':
    return ((f1 - f2)**2).sum(dim=1).sqrt().mean()


def latent_similarity_n(f, type='cost'):
  """ computes a pairwise similarity loss of n vectors
  Args:
    f: list of `n` [B, C, H, W]. `n`can be 2, 3, 4, 6.
  Returns:
    scalar float
  """
  n = len(f)
  if n == 2:
    return latent_similarity(f[0], f[1], type=type)
  elif n == 3:
    return latent_similarity_3(f[0], f[1], f[2], type=type)
  elif n == 4:
    return latent_similarity_4(f[0], f[1], f[2], f[3], type=type)
  elif n == 6:
    return latent_similarity_6(f[0], f[1], f[2], f[3], f[4], f[5], type=type)
  elif n == 12:
    return latent_similarity_12(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], type=type)


def latent_similarity_3(f1, f2, f3, type='cost'):
  """ computes a pairwise similarity loss of three vectors
  Args:
    f1: [B, C, H, W]
    f2: [B, C, H, W]
    f3: [B, C, H, W]
  Returns:
    scalar float
  """
  return (latent_similarity(f1, f2, type) +\
          latent_similarity(f1, f3, type) +\
          latent_similarity(f2, f3, type) / 3)


def latent_similarity_4(f1, f2, f3, f4, type='cost'):
  """ computes a pairwise similarity loss of four vectors
  Args:
    f1: [B, C, H, W]
    f2: [B, C, H, W]
    f3: [B, C, H, W]
    f4: [B, C, H, W]
  Returns:
    scalar float
  """
  return (latent_similarity(f1, f2, type) +\
          latent_similarity(f1, f3, type) +\
          latent_similarity(f1, f4, type) +\
          latent_similarity(f2, f3, type) +\
          latent_similarity(f2, f4, type) +\
          latent_similarity(f3, f4, type) / 6)


def latent_similarity_6(f1, f2, f3, f4, f5, f6, type='cost'):
  """ computes a pairwise similarity loss of six vectors
  Args:
    f1: [B, C, H, W]
    f2: [B, C, H, W]
    f3: [B, C, H, W]
    f4: [B, C, H, W]
    f5: [B, C, H, W]
    f6: [B, C, H, W]
  Returns:
    scalar float
  """
  return (latent_similarity(f1, f2, type) +\
          latent_similarity(f1, f3, type) +\
          latent_similarity(f1, f4, type) +\
          latent_similarity(f1, f5, type) +\
          latent_similarity(f1, f6, type) +\
          latent_similarity(f2, f3, type) +\
          latent_similarity(f2, f4, type) +\
          latent_similarity(f2, f5, type) +\
          latent_similarity(f2, f6, type) +\
          latent_similarity(f3, f4, type) +\
          latent_similarity(f3, f5, type) +\
          latent_similarity(f3, f6, type) +\
          latent_similarity(f4, f5, type) +\
          latent_similarity(f4, f6, type) +\
          latent_similarity(f5, f6, type) / 15)


def latent_similarity_12(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, type='cost'):
  """ computes a pairwise similarity loss of six vectors
  Args:
    f1: [B, C, H, W]
    f2: [B, C, H, W]
    f3: [B, C, H, W]
    f4: [B, C, H, W]
    f5: [B, C, H, W]
    f6: [B, C, H, W]
    f7: [B, C, H, W]
    f8: [B, C, H, W]
    f9: [B, C, H, W]
    f10: [B, C, H, W]
    f11: [B, C, H, W]
  Returns:
    scalar float
  """
  return (latent_similarity(f1, f2, type) +\
          latent_similarity(f1, f3, type) +\
          latent_similarity(f1, f4, type) +\
          latent_similarity(f1, f5, type) +\
          latent_similarity(f1, f6, type) +\
          latent_similarity(f1, f7, type) +\
          latent_similarity(f1, f8, type) +\
          latent_similarity(f1, f9, type) +\
          latent_similarity(f1, f10, type) +\
          latent_similarity(f1, f12, type) +\
          latent_similarity(f2, f3, type) +\
          latent_similarity(f2, f4, type) +\
          latent_similarity(f2, f5, type) +\
          latent_similarity(f2, f6, type) +\
          latent_similarity(f2, f7, type) +\
          latent_similarity(f2, f8, type) +\
          latent_similarity(f2, f9, type) +\
          latent_similarity(f2, f10, type) +\
          latent_similarity(f2, f11, type) +\
          latent_similarity(f2, f12, type) +\
          latent_similarity(f3, f4, type) +\
          latent_similarity(f3, f5, type) +\
          latent_similarity(f3, f6, type) +\
          latent_similarity(f3, f7, type) +\
          latent_similarity(f3, f8, type) +\
          latent_similarity(f3, f9, type) +\
          latent_similarity(f3, f10, type) +\
          latent_similarity(f3, f11, type) +\
          latent_similarity(f3, f12, type) +\
          latent_similarity(f4, f5, type) +\
          latent_similarity(f4, f6, type) +\
          latent_similarity(f4, f7, type) +\
          latent_similarity(f4, f8, type) +\
          latent_similarity(f4, f9, type) +\
          latent_similarity(f4, f10, type) +\
          latent_similarity(f4, f11, type) +\
          latent_similarity(f4, f12, type) +\
          latent_similarity(f5, f6, type) +\
          latent_similarity(f5, f7, type) +\
          latent_similarity(f5, f8, type) +\
          latent_similarity(f5, f9, type) +\
          latent_similarity(f5, f10, type) +\
          latent_similarity(f5, f11, type) +\
          latent_similarity(f5, f12, type) +\
          latent_similarity(f6, f7, type) +\
          latent_similarity(f6, f8, type) +\
          latent_similarity(f6, f9, type) +\
          latent_similarity(f6, f10, type) +\
          latent_similarity(f6, f11, type) +\
          latent_similarity(f6, f12, type) +\
          latent_similarity(f7, f8, type) +\
          latent_similarity(f7, f9, type) +\
          latent_similarity(f7, f10, type) +\
          latent_similarity(f7, f11, type) +\
          latent_similarity(f7, f12, type) +\
          latent_similarity(f8, f9, type) +\
          latent_similarity(f8, f10, type) +\
          latent_similarity(f8, f11, type) +\
          latent_similarity(f8, f12, type) +\
          latent_similarity(f9, f10, type) +\
          latent_similarity(f9, f11, type) +\
          latent_similarity(f9, f12, type) +\
          latent_similarity(f10, f11, type) +\
          latent_similarity(f10, f12, type) +\
          latent_similarity(f11, f12, type) / 65)


def charbonnier_loss(x, y, eps=0.001, alpha=0.5):
  """ Generalized Charbonnier Loss
  """
  if alpha == 0.5:
    return ((x - y)**2 + eps**2).sqrt().mean()


class ImageGradient():
  """ Class to compute image gradients using a 2D conv layer.
  """

  def __init__(self, device):
    kernel = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    self.kernel = torch.FloatTensor(kernel).view([1, 1, 3, 3]).to(device) / 8

  def gradient(self, x):
    """
    Args:
      x: [B,1, H, W]
    Returns:
      [B, 1, H-2, W-2]
    """
    grad_x = F.conv2d(x, self.kernel)
    grad_y = F.conv2d(x, self.kernel.permute(0, 1, 3, 2))
    return (grad_x**2 + grad_y**2).sqrt()

  def grad_x(self, x):
    """ dx
    Args:
      x: [B, 1, H, W]
    Returns:
      [B, 1, H-2, W-2]
    """
    return F.conv2d(x, self.kernel)

  def grad_x_2(self, x):
    """ dx for two input channels
    Args:
      x: [B, 2, H, W]
    Returns:
      [B, 2, H-2, W-2]
    """
    g1 = F.conv2d(x[:, 0].unsqueeze(1), self.kernel)
    g2 = F.conv2d(x[:, 1].unsqueeze(1), self.kernel)
    return torch.cat([g1, g2], dim=1)

  def grad_y(self, x):
    """ dy
    Args:
      x: [B, 1, H, W]
    Returns:
      [B, 1, H-2, W-2]
    """
    return  F.conv2d(x, self.kernel.permute(0, 1, 3, 2))

  def grad_y_2(self, x):
    """ dy for two input channels
    Args:
      x: [B, 2, H, W]
    Returns:
      [B, 2, H-2, W-2]
    """
    g1 = F.conv2d(x[:, 0].unsqueeze(1), self.kernel.permute(0, 1, 3, 2))
    g2 = F.conv2d(x[:, 1].unsqueeze(1), self.kernel.permute(0, 1, 3, 2))
    return torch.cat([g1, g2], dim=1)


def edge_aware_smoothness(image, flow, lambda_param=150, order=1, image_gradient=None):
  """ Edge aware smoothness loss for single flow.
  Args:
    image: [B, 1, H ,W], the image to warp to.
    flow:  [B, 2, H, W], the flow to `image`.
  Returns:
    scalar float
  """
  if image_gradient is None:
    image_gradient = ImageGradient(device=image.device)
  im_dx = image_gradient.grad_x(image)
  im_dy = image_gradient.grad_y(image)
  flow_dx = image_gradient.grad_x_2(flow)
  flow_dy = image_gradient.grad_y_2(flow)
  for _ in range(order - 1):
    flow_dx = image_gradient.grad_x_2(flow_dx)
    flow_dy = image_gradient.grad_y_2(flow_dy)
  return (torch.exp(-lambda_param * im_dx.abs()) * flow_dx.abs() + \
          torch.exp(-lambda_param * im_dy.abs()) * flow_dy.abs()).mean()


def edge_aware_smoothness_n(images, flows, lambda_param=150, order=1, taps=1):
  """ Edge aware smoothness loss for multiple flows.
  Args:
    image: [B, C, H ,W], the image to warp.
    flows:  [B, N, H, W], the flows to `image`.
  Returns:
    scalar float
  """
  image_gradient = ImageGradient(device=images.device)
  B, N, H, W = flows.shape
  N = N // 2

  aggregate_loss = []
  flows = flows.view(B, N, 2, H, W)
  images = images.unsqueeze(dim=2)

  if taps == 1:
    for n in range(N):
      aggregate_loss.append(edge_aware_smoothness(images[:, n], flows[:, n], image_gradient=image_gradient))
  elif taps == 2:
    images = twotap_reordering(images)
    for n in range(N):
      aggregate_loss.append(edge_aware_smoothness(images[:, 2 * n], flows[:, n], image_gradient=image_gradient))
      aggregate_loss.append(edge_aware_smoothness(images[:, 2 * n + 1], flows[:, n], image_gradient=image_gradient))
  elif taps == 4:
    for n in range(N):
      aggregate_loss.append(edge_aware_smoothness(images[:, 4 * n], flows[:, n], image_gradient=image_gradient))
      aggregate_loss.append(edge_aware_smoothness(images[:, 4 * n + 1], flows[:, n], image_gradient=image_gradient))
      aggregate_loss.append(edge_aware_smoothness(images[:, 4 * n + 2], flows[:, n], image_gradient=image_gradient))
      aggregate_loss.append(edge_aware_smoothness(images[:, 4 * n + 3], flows[:, n], image_gradient=image_gradient))

  return torch.mean(torch.stack(aggregate_loss))


def edge_aware_smoothness_n_old(image, flows, lambda_param=150, order=1, **kwargs):
  """ Edge aware smoothness loss for multiple flows.
  Args:
    image: [B, 1, H ,W], the image to warp to.
    flows:  [B, N, H, W], the flows to `image`.
  Returns:
    scalar float
  """
  image = image[:, -1].unsqueeze(dim=1)
  image_gradient = ImageGradient(device=image.device)
  _, N, _, _ = flows.size()
  N = N // 2
  im_dx = image_gradient.grad_x(image)
  im_dy = image_gradient.grad_y(image)

  flow_dx = []
  flow_dy = []
  for n in range(N):
    curr_dx = image_gradient.grad_x_2(flows[:, 2 * n:2 * (n + 1)])
    curr_dy = image_gradient.grad_y_2(flows[:, 2 * n:2 * (n + 1)])
    for _ in range(order - 1):
      curr_dx = image_gradient.grad_x_2(curr_dx)
      curr_dy = image_gradient.grad_y_2(curr_dy)
    flow_dx.append(curr_dx)
    flow_dy.append(curr_dy)
  flow_dx = torch.stack(flow_dx, dim=0)
  flow_dy = torch.stack(flow_dy, dim=0)

  return (torch.exp(-lambda_param * im_dx.abs()) * flow_dx.abs().sum(dim=0) + \
          torch.exp(-lambda_param * im_dy.abs()) * flow_dy.abs().sum(dim=0)).mean()


def edge_aware_loss_n(corrs_warped, corrs_static, masks, s=1e2):
  """ Total variation loss with an edge aware weighting for multiple frequencies.
  Applies loss on edges of intensity images.
  Args:
    pred: [B, C, H ,W], the prediction.
    target:  [B, C, H, W], the target to compute edge aware weights from.
    mask:  [B, F, H, W], to mask values
  Returns:
    scalar float
  """
  _, F, _, _, = corrs_static.shape

  image_gradient = ImageGradient(device=corrs_warped.device)
  if F == 4:
    intensity_warped = corrs_warped.mean(dim=1, keepdim=True)
    intensity_static = corrs_static.mean(dim=1, keepdim=True)
    return edge_aware_loss(intensity_warped, intensity_static, masks, s=s, image_gradient=image_gradient)
  elif F == 12:
    intensity_warped_1 = corrs_warped[:, :4].mean(dim=1, keepdim=True)
    intensity_static_1 = corrs_static[:, :4].mean(dim=1, keepdim=True)
    intensity_warped_2 = corrs_warped[:, 4:8].mean(dim=1, keepdim=True)
    intensity_static_2 = corrs_static[:, 4:8].mean(dim=1, keepdim=True)
    intensity_warped_3 = corrs_warped[:, 8:].mean(dim=1, keepdim=True)
    intensity_static_3 = corrs_static[:, 8:].mean(dim=1, keepdim=True)

    return (edge_aware_loss(intensity_warped_1, intensity_static_1, masks[:, 0].unsqueeze(dim=1), s=s, image_gradient=image_gradient) + \
            edge_aware_loss(intensity_warped_2, intensity_static_2, masks[:, 1].unsqueeze(dim=1), s=s, image_gradient=image_gradient) + \
            edge_aware_loss(intensity_warped_3, intensity_static_3, masks[:, 2].unsqueeze(dim=1), s=s, image_gradient=image_gradient)) / 3


def edge_aware_loss(pred, target, mask, s=1e-2, image_gradient=None):
  """ Total variation loss with an edge aware weighting.
  Applies loss on edges.
  Args:
    pred: [B, 1, H ,W], the prediction.
    target:  [B, 1, H, W], the target to compute edge aware weights from.
    mask:  [B, 1, H, W], to mask values
  Returns:
    scalar float
  """
  if image_gradient is None:
    image_gradient = ImageGradient(device=pred.device)
  pred_dx = image_gradient.grad_x(pred) * mask[:, :, 1:-1, 1:-1]
  pred_dy = image_gradient.grad_y(pred) * mask[:, :, 1:-1, 1:-1]
  target_dx = image_gradient.grad_x(target) * mask[:, :, 1:-1, 1:-1]
  target_dy = image_gradient.grad_y(target) * mask[:, :, 1:-1, 1:-1]
  return (torch.exp(-1 / (s + target_dx.abs())) * 1 / (s + pred_dx.abs()) + \
          torch.exp(-1 / (s + target_dy.abs())) * 1 / (s + pred_dy.abs())).mean()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Portion of this code copyright (c) 2022 Visual Computing group
                of Ulm University, Germany. See the LICENSE file at the
                top-level directory of this distribution.
    Portions of this code are based on code by Jinwei Gu and Zhile Ren.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
from torch import nn


def twotap_reordering(inputs):
  time_steps = inputs.shape[1] // 2
  if time_steps == 2:
    inputs = torch.cat([inputs[:, [0, 2]], inputs[:, [1, 3]]], dim=1)
  elif time_steps == 6:
    inputs = torch.cat([inputs[:, [0, 2]], inputs[:, [1, 3]],
                        inputs[:, [4, 6]], inputs[:, [5, 7]],
                        inputs[:, [8, 10]], inputs[:, [9, 11]]], dim=1)
  return inputs


def warp(img, flow):
  """
  warp an image/tensor according to the optical flow
  Args:
    img: [B, C, H, W], image to warp.
    flow: [B, 2, H, W], flow to be applied.
  Returns:
    output: [B, C, H ,W], warped image, invalid pixels are masked with zero.
    mask: [B, C, H, W], mask (valid pixels == 1).
  """
  B, C, H, W = img.size()
  # mesh grid with pixel ids
  xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
  yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
  xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
  yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
  grid = torch.cat((xx, yy), 1).float()  # [B, 2, H, W]

  if img.is_cuda:
    grid = grid.to(img)
  vgrid = grid + flow

  # scale grid to [-1,1], pixel locations normalized by the img spatial dimensions
  vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
  vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

  # warp image
  vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]
  output = nn.functional.grid_sample(img, vgrid, align_corners=True)

  # mask for valid warped pixels
  mask = torch.ones(img.size()).to(dtype=img.dtype, device=img.device)
  mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

  mask[mask < 0.9999] = 0
  mask[mask > 0] = 1

  return output * mask, mask


def warp_correlations(corrs, flows):
  """
  Warp correlation images/tensors according to the optical flows.
  Args:
    corrs: [B, 3, H, W], stack of three images to warp.
    flows: [B, 6, H, W] stack of three optical flows to apply.
  Returns:
    output: [B, 3, H, W], warped correlations, masked at invalid.
    mask: [B, 3, H, W], mask (valid pixels == 1).
  """
  B, _, H, W = corrs.size()
  output1, mask1 = warp(corrs[:, 0].view(B, 1, H, W), flows[:, [0, 1]])
  output2, mask2 = warp(corrs[:, 1].view(B, 1, H, W), flows[:, [2, 3]])
  output3, mask3 = warp(corrs[:, 2].view(B, 1, H, W), flows[:, [4, 5]])
  output = torch.stack((output1, output2, output3), dim=1)
  mask = torch.stack((mask1, mask2, mask3), dim=1)

  return output, mask


def warp_correlations_n(corrs, flows, taps=1):
  """
  Warp correlation images/tensors according to the optical flows.
  Args:
    corrs: [B, N, H, W], stack of `N` images to warp. (can be `N+1`)
    flows: [B, 2N, H, W] stack of `N` optical flows to apply.
    taps: sensor taps, can be `[1, 2, 4]`.
  Returns:
    output: [B, N, H, W], warped correlations, masked to zero at invalid.
    mask: [B, N, H, W], mask (valid pixels == 1).
  """
  B, N, H, W = flows.size()
  N = N // 2
  if taps == 2:
    corrs = twotap_reordering(corrs)

  outputs = []
  masks = []
  for n in range(N):
    output, mask = warp(corrs[:, n * taps:(n + 1) * taps].view(B, taps, H, W), flows[:, [2 * n, 2 * n + 1]])
    outputs.append(output)
    masks.append(mask)
  # if flow for last one is missing
  if corrs.size()[1] // taps == N + 1:
    outputs.append(corrs[:, -taps:])
    masks.append(torch.ones_like(masks[0]))
  output = torch.cat(outputs, dim=1)
  mask = torch.cat(masks, dim=1)

  if taps == 2:
    output = twotap_reordering(output)
    mask = twotap_reordering(mask)

  return output, mask


def combine_masks(masks, eps=0.01, C=4, reduce_all=False):
  """ combines masks of multiple frequencies
  Args:
    masks: [B, N, H, W], where N is divisible by `C`.
    C: number of masks to combine.
  Returns:
    [B, F, H, W], where F = N // C, or F = 1 if `reduce_all`.
  """
  B, N, H, W = masks.size()
  if not reduce_all:
    F = N // C
  else:
    F = 1
    C = N
  masks = masks.view(B, F, C, H, W)
  mask = torch.sum(masks, dim=2, keepdim=False)
  mask[mask < C - eps] = 0
  mask[mask >= C - eps] = 1
  return mask

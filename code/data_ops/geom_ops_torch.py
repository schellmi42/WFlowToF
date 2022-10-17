'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
import numpy as np

"""
These are all torch functions.
"""


# lightspeed in nanoseconds
constants = {'lightspeed': 0.299792458}
# in GHz
constants['frequencies'] = np.array([20, 50, 70]) / 1e3
constants['frequencies_str'] = ['20MHz', '50MHz', '70MHz']
# in radians
constants['phase_offsets'] = [0, np.pi / 2, np.pi, 3 / 2 * np.pi]
# in nanoseconds
constants['exposure_time'] = 0.01 / constants['lightspeed']


def correlation2depth(correlations, frequency, eps=1e-6):
  """ Computes ToF depth from intensity images. (in meter [m])
    Loops around at `1/2 * f`
    Optimized version for four phase measurements at 90° step offsets.
  Args:
    correlations: `floats` of shape `[B, 4, H, W]`.
      ordered with offsets `[0°, 90°, 180°, 270°]`
    frequency: `float` in GHz
  Returns:
    `floats` of shape `[B, H, W]`
  """
  # phase offset on light path
  delta_phi = torch.atan2(correlations[:, 3] - correlations[:, 1], correlations[:, 0] - correlations[:, 2] + eps)
  # resolve to positive domain
  delta_phi[delta_phi < 0] += 2 * np.pi
  #print('phase ', delta_phi)
  tof_depth = constants['lightspeed'] / (4 * np.pi * frequency) * delta_phi
  return torch.unsqueeze(tof_depth, dim=1)


def correlation2depth_n(correlations, frequencies, eps=1e-6):
  """ Computes ToF depth from intensity images. (in meter [m])
    Loops around at `1/2 * f`
    Optimized version for four phase measurements at 90° step offsets.
  Args:
    correlations: `floats` of shape `[B, 4*F, H, W]`.
      ordered with offsets `[0°, 90°, 180°, 270°]`
    frequencies: [F], `float` in GHz
  Returns:
    `floats` of shape `[B, F, H, W]`
  """
  if not torch.is_tensor(correlations):
    correlations = torch.tensor(correlations)
  frequencies = torch.tensor(frequencies).to(correlations.device)
  B, N, H, W = correlations.size()
  F = N // 4
  correlations = correlations.view(B, F, 4, H, W)

  # increase numerical stability with epsilon shifts, for both y and x due to internal torch implementation of atan2
  signx = torch.sign(correlations[:, :, 0] - correlations[:, :, 2])
  signx[signx == 0] = 1

  signy = torch.sign(correlations[:, :, 3] - correlations[:, :, 1])
  signy[signy == 0] = 1

  # phase offset on light path [B, N, H, W]
  delta_phi = torch.atan2(correlations[:, :, 3] - correlations[:, :, 1] + signy * eps, correlations[:, :, 0] - correlations[:, :, 2] + signx * eps)
  # resolve to positive domain
  delta_phi[delta_phi < 0] += 2 * np.pi
  #print('phase ', delta_phi)
  tof_depth = constants['lightspeed'] / (4 * np.pi * frequencies.view(1, F, 1, 1)) * delta_phi
  return tof_depth


def phase_unwrapping(tof_depths, max_length_torch):
    diff = tof_depths[:, :1] - tof_depths
    # pu_mask_neg = diff < -0.5 * max_length_torch.view(1, -1, 1, 1)
    pu_mask_3 = diff > 2.5 * max_length_torch.view(1, -1, 1, 1)
    pu_mask_2 = (diff > 1.5 * max_length_torch.view(1, -1, 1, 1))  # * torch.logical_not(pu_mask_3)
    pu_mask_1 = (diff > 0.5 * max_length_torch.view(1, -1, 1, 1))  # * torch.logical_not(pu_mask_2) * torch.logical_not(pu_mask_3)
    # tof_depths = tof_depths - pu_mask_neg.to(bool) * max_length_torch.view(1, -1, 1, 1)
    tof_depths = tof_depths + pu_mask_1.to(bool) * max_length_torch.view(1, -1, 1, 1)
    tof_depths = tof_depths + pu_mask_2.to(bool) * max_length_torch.view(1, -1, 1, 1)
    tof_depths = tof_depths + pu_mask_3 * max_length_torch.view(1, -1, 1, 1)
    return tof_depths


def phase_unwrapping_iterative(tof_depths, max_length_torch):
    tof_depths0 = tof_depths[:, 0]
    tof_depths1 = tof_depths[:, 1]
    tof_depths2 = tof_depths[:, 2]
    # step 1
    diff = tof_depths0 - tof_depths1
    # pu_mask_neg = diff < -0.5 * max_length_torch.view(1, -1, 1, 1)
    pu_mask_3 = diff > 2.5 * max_length_torch[1].view(1, 1, 1)
    pu_mask_2 = (diff > 1.5 * max_length_torch[1].view(1, 1, 1))
    pu_mask_1 = (diff > 0.5 * max_length_torch[1].view(1, 1, 1))
    # tof_depths = tof_depths - pu_mask_neg.to(bool) * max_length_torch.view(1, -1, 1, 1)
    tof_depths1 = tof_depths1 + pu_mask_1.to(bool) * max_length_torch[1].view(1, 1, 1)
    tof_depths1 = tof_depths1 + pu_mask_2.to(bool) * max_length_torch[1].view(1, 1, 1)
    tof_depths1 = tof_depths1 + pu_mask_3 * max_length_torch[1].view(1, 1, 1)

    # step 2
    diff = tof_depths1 - tof_depths2
    # pu_mask_neg = diff < -0.5 * max_length_torch.view(1, -1, 1, 1)
    # pu_mask_3 = diff > 2.5 * max_length_torch[2].view(1, 1, 1)
    pu_mask_2 = (diff > 1.5 * max_length_torch[2].view(1, 1, 1))
    pu_mask_1 = (diff > 0.5 * max_length_torch[2].view(1, 1, 1))
    # tof_depths = tof_depths - pu_mask_neg.to(bool) * max_length_torch.view(1, -1, 1, 1)
    tof_depths2 = tof_depths2 + pu_mask_1.to(bool) * max_length_torch[2].view(1, 1, 1)
    tof_depths2 = tof_depths2 + pu_mask_2.to(bool) * max_length_torch[2].view(1, 1, 1)
    # tof_depths2 = tof_depths2 + pu_mask_3 * max_length_torch[2].view(1, 1, 1)
    return torch.stack([tof_depths0, tof_depths1, tof_depths2], dim=1)

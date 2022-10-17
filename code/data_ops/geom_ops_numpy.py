'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np

"""
These are all numpy functions.
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


def _max_length(freq):
  return constants['lightspeed'] / (2 * freq)


def tof_depth_from_single_frequency(correlations, frequency, phase_offsets=[0, 120, 240]):
  """ Computes depth from single frequency measurements. Contains phase wrapping.
  Args:
    correlations: `float` of shape `[B, H, W, P]`.
    frequency: `float` in GHz.
    phase_offsets: `float` of shape '[P]` in degree.
  Returns:
    `float` of shape `[B, H, W]`.
  """
  phase_offsets = (np.array(phase_offsets) / 180 * np.pi).reshape([1, 1, 1, -1])
  I = np.sum(-np.sin(phase_offsets) * correlations, axis=-1)
  Q = np.sum(np.cos(phase_offsets) * correlations, axis=-1)
  delta_phi = np.arctan2(I, Q)
  # resolve to positive domain
  delta_phi[delta_phi < 0] += 2 * np.pi
  #print('phase ', delta_phi)
  depth = constants['lightspeed'] / (4 * np.pi * frequency) * delta_phi
  return depth


def correlation2depth(correlations, frequency):
  """ Computes ToF depth from intensity images. (in meter [m])
    Loops around at `1/2 * f`
    Optimized version for four phase measurements at 90° step offsets.
  Args:
    correlations: `floats` of shape `[B, H, W, 4]`.
      ordered with offsets `[0°, 90°, 180°, 270°]`
    frequency: `float` in GHz
  Returns:
    `floats` of shape `[B, H, W, 1]`
  """
  # phase offset on light path
  delta_phi = np.arctan2(correlations[:, :, :, 3] - correlations[:, :, :, 1], correlations[:, :, :, 0] - correlations[:, :, :, 2])
  # resolve to positive domain
  delta_phi[delta_phi < 0] += 2 * np.pi
  #print('phase ', delta_phi)
  tof_depth = constants['lightspeed'] / (4 * np.pi * frequency) * delta_phi
  return np.expand_dims(tof_depth, axis=-1)


def amplitude_and_intensity_from_correlation(corr):
  """ Computes amplitude and intensities for correlations measured at [0, 90, 180, 270] degrees.
  Args:
    corr: leading dimension is `4`.
  Returns:
    amp:  shape of corr, except for first dimension.
    int: shape of corr, except for first dimension.
  """
  amp = 0.5 * np.sqrt((corr[0] - corr[2])**2 + (corr[1] - corr[3])**2)
  int = np.sum(corr, axis=0) / 4
  return amp, int


def amplitude_and_intensity_from_correlationv2(correlations, phase_offsets):
  """
  Args:
    correlations: shape `[B, H, W, P]`
  Returns:
  amplitudes: shape `[B, H, W]`
  intensities: shape `[B, H, W]`
  """
  phase_offsets = np.reshape(phase_offsets, [1, 1, 1, -1])
  I = np.sum(np.sin(phase_offsets) * correlations, axis=3)
  Q = np.sum(np.cos(phase_offsets) * correlations, axis=3)
  amplitudes = 0.5 * np.sqrt(I**2 + Q**2)
  intensities = np.sum(correlations, axis=3) / len(phase_offsets)
  return amplitudes, intensities

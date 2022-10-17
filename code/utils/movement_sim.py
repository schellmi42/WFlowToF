'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from torch_utils.warp import warp


def jitter(data, max_pixel=2, random=True):
  """ Add random camera jittering/movement (last one stays fixed)
  Crops output images.
  Args:
    data: shape `[B, H, W, C]`
    max_pixel: positive `int`, maximal allowed movement +/-.
  Returns:
    shape `[B, H_new, W_new, N]`
  """
  N = data[0].shape[3]
  if random:
    mx, my = np.random.choice(np.arange(-max_pixel, 0), size=[2, N - 1])
    signx, signy = np.random.choice([0, 1], size=[2, N - 1])
  else:
    mx, my = -np.ones([N - 1], dtype=int) * max_pixel, -np.ones([N - 1], dtype=int) * max_pixel
    signx, signy = np.zeros(shape=[2, N - 1])
  # print(mx * (2 * signx - 1), my * (2 * signy - 1))
  moved_data = []
  for d in data:
    moved_d = []
    for n in range(N - 1):
      if signx[n]:
        d[:, :, :, n] = d[:, ::-1, :, n]
      if signy[n]:
        d[:, :, :, n] = d[:, :, ::-1, n]
      moved_d.append(d[:, -mx[n]:-mx[n] - max_pixel - 1, -my[n]:-my[n] - max_pixel - 1, n])
      if signx[n]:
        moved_d[-1][:, :, :] = moved_d[-1][:, ::-1, :]
      if signy[n]:
        moved_d[-1][:, :, :] = moved_d[-1][:, :, ::-1]
    moved_d.append(d[:, :- max_pixel - 1, :- max_pixel - 1, -1])
    moved_data.append(np.stack(moved_d, axis=-1))
  return np.stack(moved_data, axis=0)


def translation(data, max_pixel=4, random=True, signs=None, taps=1):
  """ Add linear movement in N frames (last one stays fixed)
  Crops output images.
  Args:
    data: shape `[B, H, W, N]`
    max_pixel: positive `int`, maximal allowed movement per frame +/-.
  Returns:
    shape `[B, H, W, N]`, zero padded
    flow: `[B, H, W, N, 2]`
  """
  N = data.shape[3]
  if random:
    mx, my = np.random.choice(np.arange(0, max_pixel + 1), size=2)
    # mx = 0
    signx, signy = np.random.choice([-1, 1], size=2)
  else:
    mx, my = max_pixel, max_pixel // 2
    signx, signy = 1, 1
  if signs is not None:
    signx, signy = signs
  # print(mx * signx, my * signy)
  mxs = mx * np.arange(-N // taps + 1, 1)
  mys = my * np.arange(-N // taps + 1, 1)
  if taps == 4:
    mxs = np.repeat(mxs, 4)
    mys = np.repeat(mys, 4)
  elif taps == 2:
    if len(mxs) == 2:
      mxs = mxs[[0, 1, 0, 1]]
      mys = mys[[0, 1, 0, 1]]
    elif len(mxs) == 6:
      mxs = mxs[[0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]]
      mys = mys[[0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]]
  # print(mxs, mys)
  # print(mx * (2 * signx - 1), my * (2 * signy - 1))
  moved_d = []
  if signx == -1:
    data = data[:, ::-1]
  if signy == -1:
    data = data[:, :, ::-1]
  moved_d.append(
      np.pad(data[:, -mxs[0]:-1, -mys[0]:-1, 0], ((0, 0), (0, -mxs[0] + 1), (0, -mys[0] + 1)), 'constant')
                )
  for n in range(1, N):
    moved_d.append(
        np.pad(data[:, -mxs[n]:mxs[-n - 1] - 1, -mys[n]:mys[-n - 1] - 1, n], ((0, 0), (0, -mxs[n] - mxs[-n - 1] + 1), (0, -mys[n] - mys[-n - 1] + 1)), 'constant')
                  )
  moved_data = np.stack(moved_d, axis=-1)
  if signx == -1:
    moved_data = moved_data[:, ::-1]
  if signy == -1:
    moved_data = moved_data[:, :, ::-1]
  B, H, W, _ = moved_data.shape
  flow_x = -mxs.reshape([1, 1, 1, N]) * np.ones([B, H, W, 1]) * signx
  flow_y = -mys.reshape([1, 1, 1, N]) * np.ones([B, H, W, 1]) * signy
  return np.stack(moved_data, axis=0), np.stack([flow_y, flow_x], axis=-1)


def rotation(data, max_angle=5, random=True):
  """ Add linear angular movement around view axis in N frames (last one stays fixed)
  Args:
    data: shape `[B, H, W, N]`
    max_angle: positive `float`, maximal allowed movement per frame +/-.
  Returns:
    shape `[B, H, W, N]`, zero padded.
  """
  N = data[0].shape[3]
  if random:
    angle = np.random.uniform(low=-max_angle, high=max_angle, size=1)
  else:
    angle = float(max_angle)
  angle = angle * np.arange(-N + 1, 0)
  for d in data:
    d = torch.FloatTensor(d.copy())
    for n in range(N - 1):
      d[:, :, :, n] = TF.rotate(d[:, :, :, n], angle=angle[n], interpolation=InterpolationMode.BILINEAR, fill=0)
  return data

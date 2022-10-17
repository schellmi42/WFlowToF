'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tensorboard
import torch
import time
from ptlflow.utils.flow_utils import flow_to_rgb


CMAP_JET_MASK = copy.copy(plt.cm.jet)
CMAP_JET_MASK.set_bad(color='black')

CMAP_VIRIDIS_MASK = copy.copy(plt.cm.viridis)
CMAP_VIRIDIS_MASK.set_bad(color='black')

CMAP_CW_MASK = copy.copy(plt.cm.coolwarm)
CMAP_CW_MASK.set_bad(color='black')

PLT_CONFIG = {'dpi': 300,
              'bbox_inches': 'tight'}
PLT_ERROR_CONFIG = {'cmap': CMAP_CW_MASK}
PLT_DEPTH_CONFIG = {'cmap': CMAP_VIRIDIS_MASK}
PLT_CORR_CONFIG = {'cmap': plt.get_cmap('gist_gray')}


def _add_colorbar(ax, imsh):
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(imsh, cax=cax)


def plot_correlation_warp(corrs, warped_corrs, corrs_static, masks, fig_path=None, global_step=0, writer=None, tag='', t=0, taps=1, show=False):
  _, T, _, _ = corrs.shape
  fig = plt.figure(figsize=(25, 5))
  corrs = corrs.cpu().detach().numpy()
  warped_corrs = warped_corrs.cpu().detach().numpy()
  corrs_static = corrs_static.cpu().detach().numpy()
  masks = masks.cpu().detach().numpy()
  ax = plt.subplot(151)
  ax.imshow(corrs[0, t], **PLT_CORR_CONFIG)
  ax.set_title('Corr t=' + str(t))
  ax = plt.subplot(152)
  ax.imshow(warped_corrs[0, t], **PLT_CORR_CONFIG)
  ax.set_title('Corr t=' + str(t) + ' warp')
  ax = plt.subplot(153)
  ax.imshow(corrs_static[0, t], **PLT_CORR_CONFIG)
  ax.set_title('GT corr t=' + str(T // taps))
  ax = plt.subplot(154)
  error = warped_corrs[0, t] - corrs_static[0, t]
  m = np.max(np.abs(error))
  imsh = ax.imshow(error, **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
  ax.set_title('Error')
  _add_colorbar(ax, imsh)
  ax = plt.subplot(155)
  ax.imshow(masks[0, t])
  ax.set_title('warp mask')

  fig.tight_layout()
  if show:
    plt.show()
  elif writer is None:
    plt.savefig(fig_path + tag  + '_' + str(global_step) + '_corr' + str(t) + '.png', **PLT_CONFIG)
  else:
    writer.add_figure(tag + '_corr' + str(t), fig, global_step)
  plt.close()


def plot_all_correlations(corrs, warped_corrs, corrs_static, fig_path=None, global_step=0, writer=None, tag='', t=0, taps=1, show=False):
  _, T, _, _ = corrs.shape
  fig = plt.figure(figsize=(4 * 5, T * 5))
  corrs = corrs.cpu().detach().numpy()
  warped_corrs = warped_corrs.cpu().detach().numpy()
  corrs_static = corrs_static.cpu().detach().numpy()
  for t in range(T):
    ax = plt.subplot(4, T, t + 1)
    ax.imshow(corrs[0, t], **PLT_CORR_CONFIG)
    ax.set_title('Corr t=' + str(t))
    ax = plt.subplot(4, T, t + T + 1)
    ax.imshow(warped_corrs[0, t], **PLT_CORR_CONFIG)
    ax.set_title('Warped Corr t=' + str(t))
    ax = plt.subplot(4, T, t + 2 * T + 1)
    ax.imshow(corrs_static[0, t], **PLT_CORR_CONFIG)
    ax.set_title('GT corr t=' + str(t))
    ax = plt.subplot(4, T, 3 * T + t + 1)
    error = warped_corrs[0, t] - corrs_static[0, t]
    m = np.max(np.abs(error))
    imsh = ax.imshow(error, **PLT_ERROR_CONFIG, vmin=-m, vmax=m)
    ax.set_title('Error t=' + str(t))
    _add_colorbar(ax, imsh)

  fig.tight_layout()
  if show:
    plt.show()
  elif writer is None:
    plt.savefig(fig_path + tag + '_' + str(global_step) + '_corr' + str(t) + '.png', **PLT_CONFIG)
  else:
    writer.add_figure(tag + '_corr' + str(t), fig, global_step)
  plt.close()


def plot_depths_warp(depths_m, pred_depths, tof_depths, masks, fig_path=None, global_step=0, writer=None, tag='', show=False):
  _, F, _, _ = tof_depths.size()

  vmin, vmax = 0, torch.max(tof_depths).cpu().detach().numpy()
  fig = plt.figure(figsize=(25, 5 * F))
  depths_m = depths_m.cpu().detach().numpy()
  tof_depths = tof_depths.cpu().detach().numpy()
  pred_depths = pred_depths.cpu().detach().numpy()
  masks = masks.cpu().detach().numpy()
  for f in range(F):
    ax = plt.subplot(F, 5, f * 5 + 1)
    ax.imshow(depths_m[0, f], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
    ax.set_title('ToF w motion')
    ax = plt.subplot(F, 5, f * 5 + 2)
    ax.imshow(pred_depths[0, f], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
    ax.set_title('ToF after warp')
    ax = plt.subplot(F, 5, f * 5 + 3)
    ax.imshow(tof_depths[0, f], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
    ax.set_title('GT ToF')
    ax = plt.subplot(F, 5, f * 5 + 4)
    error = pred_depths[0, f] - tof_depths[0, f]
    # m = np.max(np.abs(error))
    imsh = ax.imshow(error, **PLT_ERROR_CONFIG, vmin=-0.4, vmax=0.4)
    ax.set_title('Error ToF after warp')
    _add_colorbar(ax, imsh)
    ax = plt.subplot(F, 5, f * 5 + 5)
    ax.imshow(masks[0, 0])
    ax.set_title('warp mask')

  fig.tight_layout()
  if show:
    plt.show()
  elif writer is None:
    plt.savefig(fig_path + tag + '_' + str(global_step) + '_depth.png', **PLT_CONFIG)
  else:
    writer.add_figure(tag + '_depth', fig, global_step)
  plt.close()


def plot_depths_denoising(tof_depths_input, tof_depths_warped, pred_depths, GT_depths, masks, fig_path=None, global_step=0, writer=None, tag='', show=False):
  F = tof_depths_input.shape[1]

  vmin, vmax = 0, torch.max(pred_depths[1]).cpu().detach().numpy()
  fig = plt.figure(figsize=(20, 15))
  tof_depths_input = tof_depths_input.cpu().detach().numpy()
  tof_depths_warped = tof_depths_warped.cpu().detach().numpy()
  GT_depths = GT_depths.cpu().detach().numpy()
  pred_depths_fine = pred_depths[0].cpu().detach().numpy()
  pred_depths_coarse = pred_depths[1].cpu().detach().numpy()
  masks = masks.cpu().detach().numpy()
  for f in range(F):
    ax = plt.subplot(3, 4, 1 + f)
    ax.imshow(tof_depths_input[0, f], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
    ax.set_title('ToF input freq. ' + str(f))
  ax = plt.subplot(3, 4, 4)
  error = tof_depths_input[0, -1] - GT_depths[0, 0]
  # m = np.max(np.abs(error))
  imsh = ax.imshow(error, **PLT_ERROR_CONFIG, vmin=-0.6, vmax=0.6)
  ax.set_title('Error on input')
  _add_colorbar(ax, imsh)

  for f in range(F):
    ax = plt.subplot(3, 4, 5 + f)
    ax.imshow(tof_depths_warped[0, f], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
    ax.set_title('ToF warped freq. ' + str(f))
  # ax = plt.subplot(3, 4, 8 + 4)
  # ax.imshow(masks[0, 0])
  # ax.set_title('warp mask')
  ax = plt.subplot(3, 4, 4 + 4)
  error = tof_depths_warped[0, -1] - GT_depths[0, 0]
  # m = np.max(np.abs(error))
  imsh = ax.imshow(error, **PLT_ERROR_CONFIG, vmin=-0.6, vmax=0.6)
  ax.set_title('Error after warping')
  _add_colorbar(ax, imsh)

  ax = plt.subplot(3, 4, 8 + 1)
  ax.imshow(pred_depths_coarse[0, 0], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
  ax.set_title('predicted coarse depth')
  ax = plt.subplot(3, 4, 8 + 2)
  ax.imshow(pred_depths_fine[0, 0], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
  ax.set_title('predicted fine depth')
  ax = plt.subplot(3, 4, 8 + 3)
  ax.imshow(GT_depths[0, 0], vmin=vmin, vmax=vmax, **PLT_DEPTH_CONFIG)
  ax.set_title('GT depth')
  ax = plt.subplot(3, 4, 8 + 4)
  error = pred_depths_fine[0, 0] - GT_depths[0, 0]
  # m = np.max(np.abs(error))
  imsh = ax.imshow(error, **PLT_ERROR_CONFIG, vmin=-0.6, vmax=0.6)
  ax.set_title('Error after denoising')
  _add_colorbar(ax, imsh)

  fig.tight_layout()
  if show:
    plt.show()
  elif writer is None:
    plt.savefig(fig_path + tag + '_' + str(global_step) + '_depth.png', **PLT_CONFIG)
  else:
    writer.add_figure(tag + '_depth', fig, global_step)
  plt.close()


def plot_depths_denoising_raw(tof_depths_input, tof_depths_warped, pred_depths, GT_depths, masks, fig_path=None, global_step=0):
  _, vmax = 0, torch.max(GT_depths[0, 0] * 1.2).cpu().detach().numpy()
  _ = plt.figure(figsize=(20, 15))
  tof_depths_input = tof_depths_input.cpu().detach().numpy()
  tof_depths_warped = tof_depths_warped.cpu().detach().numpy()
  GT_depths = GT_depths.cpu().detach().numpy()
  pred_depths_fine = pred_depths[0].cpu().detach().numpy()
  masks = masks.cpu().detach().numpy()

  error_input = tof_depths_input[0, -1] - GT_depths[0, 0]
  # tof_depths_input[0, -1][error_input>0.5 * max_depth] += max_depth
  # error_input[error_input>0.5 * max_depth] -= max_depth

  error_warped = tof_depths_warped[0, -1] - GT_depths[0, 0]
  # tof_depths_warped[0, -1][error_warped>0.5 * max_depth] += max_depth
  # error_warped[error_warped>0.5 * max_depth] -= max_depth

  error_pred = pred_depths_fine[0, 0] - GT_depths[0, 0]

  imgs = [tof_depths_input[0, -1], error_input, error_warped, error_pred, tof_depths_warped[0, -1], pred_depths_fine[0, 0], GT_depths[0, 0]]
  titles = ['ToF_input', 'Error_input', 'Error_warped', 'Error_pred', 'ToF_warped', 'Depth_pred', 'Depth_GT']
  maes = [100 * np.mean(np.abs(error_input * masks[0, 0])), -1, -1, -1, 100 * np.mean(np.abs(error_warped * masks[0, 0])), 100 * np.mean(np.abs(error_pred * masks[0, 0])), -1]
  v_mins = [0, -0.6, -0.6, -0.6, 0, 0, 0]
  v_maxs = [vmax, 0.6, 0.6, 0.6, vmax, vmax, vmax]
  cmaps = [PLT_DEPTH_CONFIG['cmap'], PLT_ERROR_CONFIG['cmap'], PLT_ERROR_CONFIG['cmap'], PLT_ERROR_CONFIG['cmap'], PLT_DEPTH_CONFIG['cmap'], PLT_DEPTH_CONFIG['cmap'], PLT_DEPTH_CONFIG['cmap']]
  mask_bools = [True for i in range(len(imgs))]
  plot_raw(imgs, titles, maes, v_mins, v_maxs, cmaps, fig_path, global_step, mask_bools, masks[0, 0], '')


def plot_flow(flow, masks, fig_path=None, global_step=0, writer=None, tag='', background='bright', show=False):
  B, N, H, W = flow.size()
  T = N // 2
  fig = plt.figure(figsize=(T * 5, 10))
  flow_max_radius = torch.linalg.norm(flow[0].view(T, 2, H, W), dim=1).max().item()
  for t in range(T):
    ax = plt.subplot(2, T, t + 1)
    rgb_flow = flow_to_rgb(flow[0, 2 * t:2 * (t + 1)], background=background, flow_max_radius=flow_max_radius).permute(1, 2, 0)
    ax.imshow(rgb_flow.cpu().detach().numpy())
    ax.set_title('Flow t=' + str(t))

    ax = plt.subplot(2, T, T + t + 1)
    ax.imshow(masks[0, t].cpu().detach().numpy())
    ax.set_title('warp mask')

  fig.tight_layout()
  if show:
    plt.show()
  elif writer is None:
    plt.savefig(fig_path + tag + '_' + str(global_step) + '_flow.png', **PLT_CONFIG)
  else:
    writer.add_figure(tag + '_flow', fig, global_step)
  plt.close()
  return


def plot_correlations_raw(corrs_motion, corrs_warped, corrs_static, fig_path=None, global_step=0):
  _, T, _, _ = corrs_motion.shape
  mae_corr_motion = (corrs_motion[0] - corrs_static[0]).abs().sum(dim=(1, 2))
  mae_corr_warped = (corrs_warped[0] - corrs_static[0]).abs().sum(dim=(1, 2))
  corrs_motion = corrs_motion.cpu().detach().numpy()
  corrs_warped = corrs_warped.cpu().detach().numpy()
  corrs_static = corrs_static.cpu().detach().numpy()

  vmax_corrs = np.max([np.max(corrs_motion[0]), np.max(corrs_static[0])])
  for t in range(T):

    vmin_corrs = -0.5 * vmax_corrs
    imgs = [corrs_motion[0, t], corrs_static[0, t], corrs_warped[0, t]]  # , error_warped[t], error_motion[t]]
    titles = ['Corr_input', 'Corr_static', 'Corr_warped']  # , 'Error_Corr_warped_' + str(t), 'Error_Corr_motion_' + str(t)]
    maes = [-1, mae_corr_motion[t], mae_corr_warped[t]]  # , -1, -1]
    v_mins = [vmin_corrs, vmin_corrs, vmin_corrs]  # , np.min]
    v_maxs = [vmax_corrs, vmax_corrs, vmax_corrs]
    cmaps = [PLT_CORR_CONFIG['cmap'], PLT_CORR_CONFIG['cmap'], PLT_CORR_CONFIG['cmap']]
    mask_bools = [False, False, False]

    plot_raw(imgs, titles, maes, v_mins, v_maxs, cmaps, fig_path, global_step, mask_bools, None, '_t=' + str(t))


def plot_depths_warp_raw(tof_motion, pred_depths, tof_depths, masks, fig_path=None, global_step=0):
  _, F, _, _ = tof_depths.size()

  vmaxs = [torch.max(tof_depths[0, f]).item() for f in range(F)]
  mae_tof_motion = 100 * ((tof_motion[0] - tof_depths[0]) * masks[0]).abs().mean(dim=(1, 2))
  mae_tof_warp = 100 * ((pred_depths[0] - tof_depths[0]) * masks[0]).abs().mean(dim=(1, 2))
  tof_motion = tof_motion.cpu().detach().numpy()
  tof_depths = tof_depths.cpu().detach().numpy()
  pred_depths = pred_depths.cpu().detach().numpy()
  masks = masks.cpu().detach().numpy()
  for f in range(F):
    error_pred = pred_depths[0] - tof_depths[0]
    error_motion = tof_motion[0] - tof_depths[0]
    imgs = [tof_motion[0, f], pred_depths[0, f], tof_depths[0, f], error_motion[f], error_pred[f]]
    titles = ['ToF_input', 'ToF_warped', 'ToF_GT', 'Error_ToF_motion', 'Error_ToF_pred']
    maes = [mae_tof_motion[f], mae_tof_warp[f], -1, -1, -1]
    v_mins = [0, 0, 0, -0.4, -0.4]
    v_maxs = [vmaxs[f], vmaxs[f], vmaxs[f], 0.4, 0.4]
    cmaps = [PLT_DEPTH_CONFIG['cmap'], PLT_DEPTH_CONFIG['cmap'], PLT_DEPTH_CONFIG['cmap'], PLT_ERROR_CONFIG['cmap'], PLT_ERROR_CONFIG['cmap']]
    mask_bools = [False, True, True, True, True]

    plot_raw(imgs, titles, maes, v_mins, v_maxs, cmaps, fig_path, global_step, mask_bools, masks[0, f], '_f=' + str(f))


def plot_raw(imgs, titles, maes, v_mins, v_maxs, cmaps, fig_path, img_id, mask_bools, mask, tag):
  fig_path = fig_path[:-1] + '_s0.6_frame10/'
  import os
  os.makedirs(fig_path, exist_ok=True)
  for img, title, mae, v_min, v_max, cmap, mask_bool in zip(imgs, titles, maes, v_mins, v_maxs, cmaps, mask_bools):
    if mask_bool:
      img[mask == 0] = np.nan
    if mae != -1:
      mae_string = 'MAE__{:.2f}'.format(mae)
    else:
      mae_string = ''
    img = np.transpose(np.squeeze(img)[::-1], (1, 0))
    filename = fig_path + str(img_id).zfill(3) + '_' + title + tag + mae_string + '.png'
    plt.imsave(fname=filename, arr=img, cmap=cmap, format='png', vmin=v_min, vmax=v_max)
    plt.close()
  filename = fig_path + str(img_id).zfill(3) + '_mask' + tag + '.png'
  if mask is not None:
    plt.imsave(fname=filename, arr=np.transpose(np.squeeze(mask)[::-1], (1, 0)), cmap=plt.get_cmap('gist_gray'), format='png')
    plt.close()


def plot_flow_raw(flow, masks, fig_path=None, global_step=0, background='bright'):
  B, N, H, W = flow.size()
  T = N // 2
  flow_max_radius = torch.linalg.norm(flow[0].view(T, 2, H, W), dim=1).max().item()
  for t in range(T):
    rgb_flow = flow_to_rgb(flow[0, 2 * t:2 * (t + 1)], background=background, flow_max_radius=flow_max_radius).permute(1, 2, 0)
    img = rgb_flow.cpu().detach().numpy()
    img = np.transpose(np.squeeze(img)[::-1], (1, 0, 2))

    title = 'Flow_t=' + str(t)
    filename = fig_path + str(global_step).zfill(3) + '_' + title + '.png'
    plt.imsave(fname=filename, arr=np.squeeze(img), format='png')
    plt.close()

    title = 'Mask_t=' + str(t)
    filename = fig_path + str(global_step).zfill(3) + '_' + title + '.png'
    mask = masks[0, t].cpu().detach().numpy()
    mask = np.transpose(np.squeeze(mask)[::-1], (1, 0))
    plt.imsave(fname=filename, arr=mask, cmap=plt.get_cmap('gist_gray'), format='png')
    plt.close()

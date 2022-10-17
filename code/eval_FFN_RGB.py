'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from os import path, makedirs

import argparse

from torch.utils.tensorboard import SummaryWriter

from RGBnets.FFNet import CustomFFNet, FFN
from torch_utils.warp import warp_correlations_n, combine_masks
from data_ops.CBdataset import CBdataset
from data_ops.geom_ops_torch import correlation2depth_n
from data_ops.geom_ops_numpy import _max_length

from torch_utils import train_utils
from utils import plotting as plt
from torch_utils import loss_f

from ptlflow.utils.flow_utils import flow_to_rgb
import time
from datetime import datetime

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='PT_sintel', help='name for the log files')
parser.add_argument('--no_TB', action='store_true', help='deactivates tensorboard')
parser.add_argument('--no_log', action='store_true', help='deactivates all logging')
parser.add_argument('--log', '--l', default='', help='use for static log directory')
parser.add_argument('--taps', default=1, type=int, help='Can use single, two or four tap sensor input')
parser.add_argument('--lindner', action='store_true', help='uses the method of lindner et al. for motion based on reconstructed intensities.')
parser.add_argument('--mult_freq', '--mf', action='store_true', help='Use all three frequencies instead of only 20MHz')
parser.add_argument('--eval_test', action='store_true')
parser.add_argument('--eval_val', action='store_true')
parser.add_argument('--plot_test', action='store_true')
parser.add_argument('--plot_raw', action='store_true', help='plot all data as separate pngs without context figures for one frame per scene')
parser.add_argument('--ckpt_dir', help='path to RGB pre-trained weights')

args = FFN.FastFlowNet.add_model_specific_args(parser).parse_args()

if args.no_log:
  args.no_TB = True

tag = 'RGB_FFN_' + args.tag
if not args.no_TB:
  if args.log == '':
    writer = SummaryWriter(log_dir='runs/data_v2/' + tag)
  else:
    writer = SummaryWriter(log_dir=args.log)
  log_path = writer.log_dir + '/'
else:
  writer = None
  if args.log == '':
    log_path = 'runs/data_v2/PWC_' + tag + '/'
  else:
    log_path = args.log

if args.log == '':
  fig_path = 'figures/data_v2/PWC_' + tag + '/'
else:
  tag = args.log.split('/')[-2]
  fig_path = args.log + '/figures/'

if not path.exists(fig_path) and ((args.no_TB and not args.no_log) or args.plot_test):
  print('[mkdir "{}"]'.format(fig_path))
  makedirs(fig_path)

if args.plot_raw:
  test_fig_path = 'figures/small_error_scale/runs/final/' + tag + '/'
  if not test_fig_path.endswith('/'):
    test_fig_path += '/'
  if not path.exists(test_fig_path):
    print('[mkdir "{}"]'.format(test_fig_path))
    makedirs(test_fig_path)


def eval(dataloader, model, loss_fn, plot, track_valid_ratio=False):
  with torch.no_grad():
    model.eval()
    losses = {'warp': [], 'tof': [], 'tof_ref': [], 'warp_ref': [], 'tof_masked': [], 'tof_masked_ref': []}
    if track_valid_ratio:
      losses['valid_ratio'] = []
    for batch, data in enumerate(dataloader):
      corrs = data['corrs']
      depths = data['depths']
      tof_depths = data['tof_depths']
      corrs_static = data['corrs_static']
      valid_masks = data['masks']
      corrs, depths, tof_depths, corrs_static, valid_masks = corrs.to(device), depths.to(device), tof_depths.to(device), corrs_static.to(device), valid_masks.to(device)
      # Compute prediction error
      output = model(corrs)
      flow = torch.cat(output['flows'], dim=1)
      # flow = torch.clip(flow, min=-50, max=50)
      warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)
      loss_warp = loss_fn(warped_corrs, corrs_static * masks)
      loss_warp_ref = loss_fn(corrs, corrs_static)

      masks_per_freq = combine_masks(masks) * valid_masks
      pred_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * masks_per_freq
      depths_m = correlation2depth_n(corrs, frequencies_GHz)
      if plot and batch == 0 and not args.no_log:
        plt.plot_correlation_warp(corrs, warped_corrs, corrs_static, masks, fig_path, epoch, writer, tag='val', taps=args.taps)
        plt.plot_flow(flow, masks, fig_path, epoch, writer, tag='val')
        plt.plot_depths_warp(depths_m, pred_depths, tof_depths, masks_per_freq, fig_path, epoch, writer, tag='val')
      loss_tof = loss_fn(pred_depths, tof_depths * masks_per_freq)
      loss_tof_ref = loss_fn(depths_m * valid_masks, tof_depths * valid_masks)

      motion_error_masks = (torch.abs(depths_m - tof_depths) > (0.4)).long()
      motion_detected = ((masks_per_freq * motion_error_masks) != 0).sum()
      if motion_detected:
        loss_tof_masked = loss_fn(pred_depths * masks_per_freq * motion_error_masks, tof_depths * masks_per_freq * motion_error_masks) * \
            np.product(depths.shape) / ((masks_per_freq * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked = torch.tensor(0)

      if motion_detected:
        loss_tof_masked_ref = loss_fn(depths_m * valid_masks * motion_error_masks, tof_depths * valid_masks * motion_error_masks) * \
            np.product(depths.shape) / ((valid_masks * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked_ref = torch.tensor(0)

      losses['warp'].append(loss_warp.item())
      losses['warp_ref'].append(loss_warp_ref.item())
      losses['tof'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      losses['tof_masked'].append(loss_tof_masked.item())
      losses['tof_masked_ref'].append(loss_tof_masked_ref.item())
      if track_valid_ratio:
        losses['valid_ratio'].append((masks_per_freq.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
    return losses


def eval_save_figs(dataloader, model, loss_fn, curr_writer=None):
  with torch.no_grad():
    model.eval()
    losses = {'warp': [], 'tof': [], 'tof_ref': [], 'valid_ratio': [], 'warp_ref': [], 'tof_masked': [], 'tof_masked_ref': []}
    i = 0
    for batch, data in enumerate(dataloader):
      print('\r' + str(i), end='')
      i += 1
      corrs = data['corrs']
      depths = data['depths']
      tof_depths = data['tof_depths']
      corrs_static = data['corrs_static']
      valid_masks = data['masks']
      corrs, depths, tof_depths, corrs_static, valid_masks = corrs.to(device), depths.to(device), tof_depths.to(device), corrs_static.to(device), valid_masks.to(device)
      # Compute prediction error
      output = model(corrs)
      flow = torch.cat(output['flows'], dim=1)
      # flow = torch.clip(flow, min=-50, max=50)
      warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)

      masks_per_freq = combine_masks(masks) * valid_masks
      pred_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * masks_per_freq
      depths_m = correlation2depth_n(corrs, frequencies_GHz)
      if not args.no_log and not args.plot_raw:
        plt.plot_correlation_warp(corrs, warped_corrs, corrs_static, masks, fig_path, batch, curr_writer, tag='val_full', taps=args.taps)
        plt.plot_flow(flow, masks, fig_path, batch, curr_writer, tag='val_full')
        plt.plot_depths_warp(depths_m, pred_depths, tof_depths, masks_per_freq, fig_path, batch, curr_writer, tag='val_full')
      if args.plot_raw:
        plt.plot_depths_warp_raw(tof_motion=depths_m, pred_depths=pred_depths, tof_depths=tof_depths, masks=masks_per_freq, fig_path=test_fig_path, global_step=i)
        # plt.plot_correlations_raw(corrs_motion=corrs, corrs_warped=warped_corrs, corrs_static=corrs_static, fig_path=test_fig_path, global_step=i)
        # plt.plot_flow_raw(flow=flow, masks=masks, fig_path=test_fig_path, global_step=i)
      loss_warp = loss_fn(warped_corrs, corrs_static * masks)
      loss_tof = loss_fn(pred_depths, tof_depths * masks_per_freq)
      loss_tof_ref = loss_fn(depths_m * valid_masks, tof_depths * valid_masks)
      loss_warp_ref = loss_fn(corrs, corrs_static)

      motion_error_masks = (torch.abs(depths_m - tof_depths) > (0.4)).long()
      motion_detected = ((masks_per_freq * motion_error_masks) != 0).sum()
      if motion_detected:
        loss_tof_masked = loss_fn(pred_depths * masks_per_freq * motion_error_masks, tof_depths * masks_per_freq * motion_error_masks) * \
            np.product(depths.shape) / ((masks_per_freq * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked = torch.tensor(0)

      if motion_detected:
        loss_tof_masked_ref = loss_fn(depths_m * valid_masks * motion_error_masks, tof_depths * valid_masks * motion_error_masks) * \
            np.product(depths.shape) / ((valid_masks * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked_ref = torch.tensor(0)

      losses['warp'].append(loss_warp.item())
      losses['warp_ref'].append(loss_warp_ref.item())
      losses['tof'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      losses['tof_masked'].append(loss_tof_masked.item())
      losses['tof_masked_ref'].append(loss_tof_masked_ref.item())
      losses['valid_ratio'].append((masks_per_freq.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
    print('\r')
  return losses

time_steps = (4 + args.mult_freq * 8) // args.taps
if args.mult_freq:
  frequencies = [20, 50, 70]
  feature_type = 'mf_c2'
else:
  frequencies = [20]
  feature_type = 'sf_c2'

frequencies_GHz = np.array(frequencies, dtype=np.float32) / 1e3
max_length = _max_length(frequencies_GHz)

if not args.lindner:
  in_channels = args.taps
else:
  if args.taps == 1:
    raise ValueError('Lindner method needs taps > 1!')
  in_channels = 1

model = CustomFFNet(args=args, in_channels=in_channels, time_steps=time_steps, taps=args.taps, convert='RGB').to(device)
# print(model)
# print(train_utils.pytorch_total_params(model))

loss = nn.L1Loss()
epoch = 0

if path.exists(args.ckpt_dir):
  checkpoint = torch.load(args.ckpt_dir)
  model.load_state_dict(checkpoint['state_dict'])
  print('[ckpt restored]' + '\n      - ' + args.ckpt_dir)

if args.mult_freq:
  results_file = 'results_' + 'MF' + str(args.taps) + '.txt'
else:
  results_file = 'results_' + 'SF' + str(args.taps) + '.txt'

if args.plot_raw:
  full_scene_in_epoch = False
else:
  full_scene_in_epoch = True

# save more figures
if args.eval_val:
  print('[final plotting on validation set]')
  dataloader_val_full = CBdataset(
      batch_size=1, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=full_scene_in_epoch, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  t1 = time.time()
  l = eval_save_figs(dataloader_val_full, model, loss, curr_writer=writer)
  t2 = time.time()
  print('val losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f}, ' for key, val in zip(l.keys(), l.values())]) + f', time: {t2 - t1:3f}')
  with open(results_file, 'a') as rFile:
    rFile.write(tag + '\nval losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(l.keys(), l.values())]) + '\n')

if args.eval_test:
  if not path.exists('test_results'):
    print('[mkdir "{}"]'.format('test_results'))
    makedirs('test_results')

  print('[evaluation on test set]')
  dataloader_test_full = CBdataset(
      batch_size=1, set='test', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=full_scene_in_epoch, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  print('Frames: {}'.format(len(dataloader_test_full)))
  t1 = time.time()
  if args.plot_test or args.plot_raw:
    l = eval_save_figs(dataloader_test_full, model, loss, curr_writer=None)
  else:
    l = eval(dataloader_test_full, model, loss, plot=False, track_valid_ratio=True)
  t2 = time.time()
  print('test losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f}, ' for key, val in zip(l.keys(), l.values())]) + f', time: {t2 - t1:3f}')
  if not args.plot_raw:
    with open('test_results/' + results_file, 'a') as rFile:
      rFile.write(tag + '\ntest losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(l.keys(), l.values())]) + '\n')

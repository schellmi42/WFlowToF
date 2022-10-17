import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from os import path, makedirs
# filter runtime warning from division by zero
import warnings
warnings.filterwarnings('ignore')
import argparse

from torch.utils.tensorboard import SummaryWriter

from CFNet.model import CoarseFineCNN
from data_ops.CBdataset import CBdataset

from torch_utils import train_utils
from utils import plotting as plt
from torch_utils import loss_f

import time
from datetime import datetime

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='SF_depth', help='name for the log files')
parser.add_argument('--main_loss', default='L1')
parser.add_argument('--aug_noise', '--noise', action='store_true')
parser.add_argument('--LRS_patience', '--lrp', default=200, type=int)
parser.add_argument('--LRS_factor', '--lrf', default=0.5, type=float)
parser.add_argument('--no_TB', action='store_true')
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--epochs', '--n', default=1000, type=int)
parser.add_argument('--log', '--l', default='')
parser.add_argument('--taps', default=1, type=int, help='Can use single, two or four tap sensor input')
parser.add_argument('--mult_freq', '--mf', action='store_true', help='Use three frequencies instead of only 20MHz')
parser.add_argument('--eval_test', action='store_true')
parser.add_argument('--eval_val', action='store_true')
parser.add_argument('--plot_test', action='store_true')
args = parser.parse_args()

if args.no_log:
  args.no_TB = True

id_string = 'SF' * (not args.mult_freq) + 'MF' * args.mult_freq + '_' + str(args.taps) + 'Tap/'
tag = 'CFN_' + args.tag + '_' + datetime.now().strftime('%b%d_%H-%M-%S')

if not args.no_TB:
  if args.log == '':
    writer = SummaryWriter(log_dir='runs/data_v2/' + id_string + tag)
  else:
    writer = SummaryWriter(log_dir=args.log)
  log_path = writer.log_dir + '/'
else:
  writer = None
  if args.log == '':
    log_path = 'runs/data_v2/' + id_string + 'CFN_' + tag + '/'
  else:
    log_path = args.log

if args.log == '':
  fig_path = 'figures/data_v2/' + id_string + 'CFN_' + tag + '/'
else:
  if not tag.endswith('/'):
    tag += '/'
  tag = args.log.split('/')[-2]
  fig_path = args.log + '/figures/'

if not path.exists(fig_path) and ((args.no_TB and not args.no_log) or args.plot_test):
  makedirs(fig_path)


def train(dataloader, model, loss_fn, optimizer, plot):
  # size = len(dataloader)
  model.train()
  losses = {'tof': [], 'tof_ref': [], 'valid_ratio': []}
  # t1 = time.time()
  for batch, data in enumerate(dataloader):
    # t2 = time.time()
    depths = data['depths'][:, 0]
    tof_depths = data['tof_depths'][:, 0]
    tof_depths_motion = data['tof_depths_motion'][:, 0]
    valid_masks = data['masks'][:, 0]
    features = tof_depths_motion
    if 'features' in data.keys():
      features = data['features'][:, 0]
    depths, tof_depths, tof_depths_motion, features, valid_masks = depths.to(device), tof_depths.to(device), tof_depths_motion.to(device), features.to(device), valid_masks.to(device)
    # Compute prediction error
    # t3 = time.time()
    pred_depths = model(features, tof_depths_motion)

    loss_tof_coarse = loss_fn(pred_depths[0] * valid_masks, tof_depths * valid_masks)
    loss_tof_fine = loss_fn(pred_depths[1] * valid_masks, tof_depths * valid_masks)
    loss_total = loss_tof_coarse + loss_tof_fine

    # Backpropagation
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    # t6 = time.time()
    # print(t2 - t1, 'data loading')
    # print(t3 - t2, 'data to GPU')
    # print(t4 - t3, 'flow prediction')
    # print(t5 - t4, 'warping')
    # print(t6 - t5, 'backprop')
    # print(t6-t2)

    loss_tof_ref = loss_fn(tof_depths_motion * valid_masks, tof_depths * valid_masks)
    losses['tof'].append(loss_tof_fine.item())
    losses['tof_ref'].append(loss_tof_ref.item())
    losses['valid_ratio'].append((valid_masks.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
    # if batch % 100 == 0:
    #   lw, lt  = np.mean(losses_warp), np.mean(losses_tof),
    #   print(f"loss: {lw:>7f}   {lt:>7f}  [{batch:>5d}/{size:>5d}]")
    # t1 = time.time()

    if plot and batch == 0 and not args.no_log:
      with torch.no_grad():
        mask = torch.zeros_like(pred_depths[0])
        plt.plot_depths_warp(tof_depths_motion, pred_depths[0], tof_depths, mask, fig_path, epoch, writer, tag='train_fine')
        plt.plot_depths_warp(tof_depths_motion, pred_depths[1], tof_depths, mask, fig_path, epoch, writer, tag='train_coarse')
  return losses


def eval(dataloader, model, loss_fn, plot):
  with torch.no_grad():
    # size = len(dataloader)
    model.eval()
    losses = {'tof': [], 'tof_ref': [], 'valid_ratio': [], 'tof_masked': []}
    # t1 = time.time()
    for batch, data in enumerate(dataloader):
      # t2 = time.time()
      depths = data['depths']
      tof_depths = data['tof_depths']
      tof_depths_motion = data['tof_depths_motion']
      valid_masks = data['masks']
      features = tof_depths_motion
      if 'features' in data.keys():
        features = data['features']
      depths, tof_depths, tof_depths_motion, features, valid_masks = depths.to(device), tof_depths.to(device), tof_depths_motion.to(device), features.to(device), valid_masks.to(device)
      # Compute prediction error
      # t3 = time.time()
      pred_depths = model(features, tof_depths_motion)

      if plot and batch == 0 and not args.no_log:
          mask = torch.zeros_like(pred_depths[0])
          plt.plot_depths_warp(tof_depths_motion, pred_depths[0], tof_depths, mask, fig_path, epoch, writer, tag='val_fine')
          plt.plot_depths_warp(tof_depths_motion, pred_depths[1], tof_depths, mask, fig_path, epoch, writer, tag='val_coarse')
      loss_tof = loss_fn(pred_depths[0] * valid_masks, tof_depths * valid_masks)

      loss_tof_ref = loss_fn(tof_depths_motion * valid_masks, tof_depths * valid_masks)
      motion_error_masks = (torch.abs(tof_depths_motion - tof_depths) > (0.4)).long()
      motion_detected = ((valid_masks * motion_error_masks) != 0).sum()
      if motion_detected:
        loss_tof_masked = loss_fn(pred_depths[0] * valid_masks * motion_error_masks, tof_depths * valid_masks * motion_error_masks) * \
            np.product(depths.shape) / ((valid_masks * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked = torch.tensor(0)

      losses['tof'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      losses['tof_masked'].append(loss_tof_masked.item())
      losses['valid_ratio'].append((valid_masks.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
      # if batch % 100 == 0:
      #   lw, lt  = np.mean(losses_warp), np.mean(losses_tof),
      #   print(f"loss: {lw:>7f}   {lt:>7f}  [{batch:>5d}/{size:>5d}]")
      # t1 = time.time()
    return losses


def eval_save_figs(dataloader, model, loss_fn, curr_writer=None):
  with torch.no_grad():
    model.eval()
    losses = {'tof': [], 'tof_ref': [], 'tof_masked': [], 'valid_ratio': []}
    for batch, data in enumerate(dataloader):
      # t2 = time.time()
      depths = data['depths']
      tof_depths = data['tof_depths']
      tof_depths_motion = data['tof_depths_motion']
      valid_masks = data['masks']
      features = tof_depths_motion
      if 'features' in data.keys():
        features = data['features']
      depths, tof_depths, tof_depths_motion, features, valid_masks = depths.to(device), tof_depths.to(device), tof_depths_motion.to(device), features.to(device), valid_masks.to(device)
      # Compute prediction error
      # t3 = time.time()
      pred_depths = model(features, tof_depths_motion)

      mask = torch.zeros_like(pred_depths[0])
      if not args.no_log:
        plt.plot_depths_warp(tof_depths_motion, pred_depths[0], tof_depths, mask, fig_path, batch, curr_writer, tag='val_full_fine')
        plt.plot_depths_warp(tof_depths_motion, pred_depths[1], tof_depths, mask, fig_path, batch, curr_writer, tag='val_full_coarse')
      loss_tof = loss_fn(pred_depths[0] * valid_masks, tof_depths * valid_masks)
      motion_error_masks = (torch.abs(tof_depths_motion - tof_depths) > (0.4)).long()
      motion_detected = ((valid_masks * motion_error_masks) != 0).sum()
      if motion_detected:
        loss_tof_masked = loss_fn(pred_depths[0] * valid_masks * motion_error_masks, tof_depths * valid_masks * motion_error_masks) * \
            np.product(depths.shape) / ((valid_masks * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked = torch.tensor(0)

      loss_tof_ref = loss_fn(tof_depths_motion * valid_masks, tof_depths * valid_masks)
      losses['tof'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      losses['tof_masked'].append(loss_tof_masked.item())
      losses['valid_ratio'].append((valid_masks.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
  return losses


time_steps = (4 + args.mult_freq * 8) / args.taps
if args.mult_freq:
  frequencies = [20, 50, 70]
  feature_type = 'mf_agresti'
  num_features = 5
else:
  frequencies = [20]
  feature_type = 'sf_c'
  num_features = 1

model = CoarseFineCNN(input_feature_size=num_features, num_output_features=len(frequencies)).to(device)
# print(model)
# print(train_utils.pytorch_total_params(model))

dataset_train = CBdataset(
    batch_size=1, set='train', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=False, shuffle=True, taps=args.taps,
    height=512, width=512, aug_rot=True, aug_noise=args.aug_noise, aug_noise_tof=args.aug_noise, aug_flip=True, aug_material=True, noise_level=0.02)
# parallel dataloader for speedup during training
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8,
                                               shuffle=False, num_workers=4)

dataset_val = CBdataset(
    batch_size=8, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=False, shuffle=True, taps=args.taps,
    height=512, width=512, aug_rot=True, aug_noise=True, aug_flip=True, noise_level=0.02)
# dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=8,
#                                              shuffle=False, num_workers=4)

loss = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.LRS_factor, patience=args.LRS_patience, verbose=True)
epoch_start = 0

if path.exists(log_path + 'checkpoint.pt'):
  model, optimizer, epoch_start = train_utils.load_ckp(log_path + 'checkpoint.pt', model, optimizer)
  # model.load_state_dict(torch.load(log_path + "model.pth"))
  print('[ckpt restored]' + '\n      - ' + log_path + 'checkpoint.pt')
  print('      - epoch: ' + str(epoch_start))
epoch_save = 50
epochs = args.epochs
for epoch in range(epoch_start, epochs):
  print(f'Epoch {epoch}/{epochs}')
  t1 = time.time()
  lt = train(dataloader_train, model, loss, optimizer, epoch % 30 == 0)
  t2 = time.time()
  print('train  loss: ' + str(np.mean(lt['tof'])) + ', time: ' + str(t2 - t1))
  if writer is not None:
    writer.add_scalar('loss_tof/train', np.mean(lt['tof']), epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

  lt = eval(dataset_val, model, loss, epoch % 30 == 0)
  print('val  loss: ' + str(np.mean(lt['tof'])))
  if writer is not None:
    writer.add_scalar('loss_tof/val', np.mean(lt['tof']), epoch)

  # if epoch%20 == 0:
  lr_scheduler.step(np.mean(lt['tof']))

  if (epoch + 1) % epoch_save == 0:
    train_utils.save_ckp({'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()},
                         log_path)
    # torch.save(model.state_dict(), log_path + "model.pth")
    print("[ckpt saved]")
  dataset_train.on_epoch_end()
  dataset_val.on_epoch_end()

if args.mult_freq:
  results_file = 'results_' + 'MF' + str(args.taps) + '.txt'
else:
  results_file = 'results_' + 'SF' + str(args.taps) + '.txt'

# save more figures
if args.epochs != 0 or args.eval_val:
  print('[final plotting on validation set]')
  dataset_val_full = CBdataset(
      batch_size=1, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=False, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  lt = eval_save_figs(dataset_val_full, model, loss, curr_writer=writer)

  print('val loss:' + str(np.mean(lt['tof'])))
  with open(results_file, 'a') as rFile:
    rFile.write(tag + '\nval losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(lt.keys(), lt.values())]) + '\n')

if args.eval_test:
  if not path.exists('test_results'):
    print('[mkdir "{}"]'.format('test_results'))
    makedirs('test_results')

  print('[evaluation on test set]')
  dataloader_test_full = CBdataset(
      batch_size=1, set='test', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=True, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  print('Frames: {}'.format(len(dataloader_test_full)))
  if args.plot_test:
    lt = eval_save_figs(dataloader_test_full, model, loss, curr_writer=None)
  else:
    lt = eval(dataloader_test_full, model, loss, plot=False)

  print('test loss:' + str(np.mean(lt['tof'])))
  with open('test_results/' + results_file, 'a') as rFile:
    rFile.write(tag + '\ntest losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(lt.keys(), lt.values())]) + '\n')

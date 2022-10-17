'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch


def pytorch_total_params(model):
  """ Computes the number of trainable parameters in a model.
  Based on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_ckp(state, checkpoint_dir, is_best=False, best_model_dir=''):
  """ To save a checkpont dict.
  Args:
    state: `dict`, checkpoint states.
    checkpoint_dir: `str`.
    is_best: `bool`, to save to separate location for best model.
    best_model_dir: `str`.
  """
  f_path = checkpoint_dir + 'checkpoint.pt'
  torch.save(state, f_path)
  if is_best:
    best_fpath = best_model_dir + 'best_model.pt'
    torch.save(state, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer=None):
  """
  Args:
    checkpoint_fpath: `str`, path to checkpoint file.
    model: state dict is loaded.
    optmizer: state dict gets loaded.
  Returns:
    model
    optimizer
    epoch
  """
  checkpoint = torch.load(checkpoint_fpath)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
  return model, optimizer, checkpoint['epoch']

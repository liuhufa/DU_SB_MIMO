# Author: PAFF
# CreatTime: 3/17/2025
# FileName: Deepunfolding SB
import os

import json
import random
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
import torch.storage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from DU_SB import DU_SB


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
class ValueWindow:

  def __init__(self, nlen=10):
    self.values: List[float] = []
    self.nlen = nlen

  def add(self, v:float):
    self.values.append(v)
    self.values = self.values[-self.nlen:]

  @property
  def mean(self):
    return sum(self.values) / len(self.values) if self.values else 0.0


def load_data(limit:int) -> List[Tuple]:
  dataset = []
  folder = 'train_dataset'
  files = sorted(os.listdir((folder)))
  N = 12
  for idx, filename in enumerate(tqdm(files)):
    if idx > limit > 0: break
    filepath = os.path.join(folder, filename)
    with open(filepath, "r") as f:
      data = json.load(f)
    h = np.zeros(N)
    J = np.zeros((N, N))
    hyperedges = data["J"]
    coeffs = data["c"]
    for edge, coef in zip(hyperedges, coeffs):
      if len(edge) == 1:
        h[edge[0]] += coef
      elif len(edge) == 2:
        i, j = edge
        J[i, j] += coef
        J[j, i] += coef
      else:
        print("warning: invalid input", edge)

    label = data.get("label", None)
    if label is not None:
      label = label.get("x", None)
    J = -J
    h = -h
    J = torch.tensor(J, dtype=torch.float32, device=device)
    h = torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(1)
    label = torch.tensor(label, dtype=torch.float32, device=device) if label is not None else None

    dataset.append((J, h, label))

  return dataset

def train(args):
  print('device:', device)
  print('hparam:', vars(args))
  exp_name = f'DU-SB_T={args.n_iter}_lr={args.lr}{"_overfit" if args.overfit else ""}'

  ''' Data '''
  dataset = load_data(args.limit)

  ''' Model '''
  model: DU_SB = globals()['DU_SB'](args.n_iter, args.batch_size).to(device)
  optim = Adam(model.parameters(), args.lr)

  ''' Ckpt '''
  init_step = 0
  losses = []
  if args.load:
    print(f'>> resume from {args.load}')
    ckpt = torch.load(args.load, map_location='cpu')
    init_step = ckpt['steps']
    losses.extend(ckpt['losses'])
    model.load_state_dict(ckpt['model'], strict=False)
    try:
      optim.load_state_dict(ckpt['optim'])
    except:
      optim_state_ckpt = ckpt['optim']
      optim_state_cur = optim.state_dict()
      optim_state_ckpt['param_groups'][0]['params'] = optim_state_cur['param_groups'][0]['params']
      optim_state_ckpt['state'] = optim_state_cur['state']
      optim.load_state_dict(optim_state_ckpt)

  ''' Bookkeep '''
  loss_wv = ValueWindow(100)
  steps_minor = 0
  steps = init_step

  ''' Train '''
  model.train()
  try:
    pbar = tqdm(total=args.steps-init_step)
    while steps < init_step + args.steps:
      if not args.no_shuffle and steps_minor % len(dataset) == 0:
        random.shuffle(dataset)
      sample = dataset[steps_minor % len(dataset)]

      J, h, bits = sample

      spins = model(J, h)
      loss_each = torch.stack([ber_loss(sp, bits, args.loss_fn) for sp in spins])
      loss = getattr(loss_each, args.agg_fn)()
      loss_for_backward: Tensor = loss / args.grad_acc
      loss_for_backward.backward()

      loss_wv.add(loss.item())

      steps_minor += 1

      if args.grad_acc == 1 or steps_minor % args.grad_acc:
        optim.step()
        optim.zero_grad()
        steps += 1
        pbar.update()

      if steps % 50 == 0:
        losses.append(loss_wv.mean)
        print(f'>> [step {steps}] loss: {losses[-1]}')
  except KeyboardInterrupt:
    pass

  ''' Ckpt '''
  ckpt = {
    'steps': steps,
    'losses': losses,
    'model': model.state_dict(),
    'optim': optim.state_dict(),
  }
  torch.save(ckpt, LOG_PATH / f'{exp_name}.pth')

  with torch.no_grad():
    params = {
      'deltas': model.Δ.detach().cpu().numpy().tolist(),
      'eta':    model.η.detach().cpu().item(),
      'a':      model.a.detach().cpu().numpy().tolist(),
    }
    print('params:', params)

    with open(LOG_PATH / f'{exp_name}.json', 'w', encoding='utf-8') as fh:
      json.dump(params, fh, indent=2, ensure_ascii=False)

  plt.plot(losses)
  plt.tight_layout()
  plt.savefig(LOG_PATH / f'{exp_name}.png', dpi=600)

def ber_loss(spins:Tensor, bits:Tensor, loss_fn:str='mse') -> Tensor:
  bits_final = (spins + 1) / 2
  if loss_fn in ['l2', 'mse']:
    return F.mse_loss(bits_final, bits)
  elif loss_fn in ['l1', 'mae']:
    return F.l1_loss(bits_final, bits)
  elif loss_fn == 'bce':
    pseudo_logits = bits_final * 2 - 1
    return F.binary_cross_entropy_with_logits(pseudo_logits, bits)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', '--n_iter', default=10, type=int)
  parser.add_argument('-B', '--batch_size', default=32, type=int, help='SB candidate batch size')
  parser.add_argument('--steps', default=30000, type=int)
  parser.add_argument('--loss_fn', default='bce', choices=['mse', 'l1', 'bce'])
  parser.add_argument('--agg_fn', default='max', choices=['mean', 'max'])
  parser.add_argument('--grad_acc', default=1, type=int, help='training batch size')
  parser.add_argument('--lr', default=1e-4, type=float)
  parser.add_argument('--load', help='ckpt to resume')
  parser.add_argument('-L', '--limit', default=-1, type=int, help='limit dataset n_sample')
  parser.add_argument('--overfit', action='store_true', help='overfit to given dataset')
  parser.add_argument('--no_shuffle', action='store_true', help='no shuffle dataset')
  parser.add_argument('--log_every', default=50, type=int)
  args = parser.parse_args()

  if args.overfit:
    print('[WARN] you are trying to overfit to the given dataset!')

  train(args)
# Author: PAFF
# CreatTime: 3/17/2025
# FileName: Deepunfolding SB

import math
import json
import random
import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torch.nn import Parameter
import torch.storage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from DU_SB import DU_SB
from numpy import ndarray
import tensorflow as tf
from sionna.mapping import Constellation, Mapper
mapper_cache: Dict[int, Mapper] = {}
constellation_cache: Dict[int, Constellation] = {}
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

def get_constellation(nbps: int) -> Constellation:
  if nbps not in constellation_cache:
    constellation_cache[nbps] = Constellation('qam', nbps)
  constellation = constellation_cache[nbps]
  # constellation.show() ; plt.show()
  return constellation
def get_mapper(nbps: int) -> Mapper:
  if nbps not in mapper_cache:
    constellation = get_constellation(nbps)
    mapper_cache[nbps] = Mapper(constellation=constellation)
  mapper = mapper_cache[nbps]
  return mapper

def modulate_and_transmit(bits: ndarray, H: ndarray, nbps: int, SNR: int = None) -> Tuple[ndarray, ndarray]:
  mapper = get_mapper(nbps)
  b = tf.convert_to_tensor(bits, dtype=tf.int32)
  x: ndarray = mapper(b).cpu().numpy()

  noise = 0
  if SNR:
    # SNR(dB) := 10*log10(P_signal/P_noise) ?= Var(signal) / Var(noise)
    sigma = np.var(bits) / SNR
    noise = np.random.normal(scale=sigma ** 0.5, size=x.shape)
  y = H @ x + noise
  return x, y

def to_ising(H:Tensor, y:Tensor, nbps:int) -> Tuple[Tensor, Tensor]:
  # the size of constellation, the M-QAM where M in {16, 64, 256}
  M = 2**nbps
  # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
  Nr, Nt = H.shape
  N = 2 * Nt
  # n_bits/n_spins that one elem decodes to
  rb = nbps // 2
  # QAM variance for normalization
  qam_var = 2 * (M - 1) / 3

  # Eq. 7 the transform matrix T from arXiv:2105.10535
  I = torch.eye(N, device=H.device)
  # [rb, N, N]
  T: Tensor = (2**(rb - 1 - torch.arange(rb, device=H.device)))[:, None, None] * I[None, ...]
  # [rb*N, N] => [N, rb*N]
  T = T.reshape(-1, N).T

  # Eq. 1
  H_tilde = torch.vstack([
    torch.hstack([H.real, -H.imag]),
    torch.hstack([H.imag,  H.real]),
  ])
  y_tilde = torch.cat([y.real, y.imag])

  H_tilde_T = H_tilde @ T
  J = -T.T @ H_tilde.T @ H_tilde @ T * (2 / qam_var)
  J = J * (1 - torch.eye(J.shape[0], device=H.device))    # mask diagonal to zeros
  z = (y_tilde - H_tilde_T @ torch.ones([N * rb, 1], device=H.device) + (math.sqrt(M) - 1) * H_tilde @ torch.ones([N, 1], device=H.device)) / math.sqrt(qam_var)
  h = 2 * z.T @ H_tilde @ T

  # [rb*N, rb*N], [rb*N, 1]
  return J, h.T

def load_data(limit:int) -> List[Tuple]:
  dataset = []
  for idx in tqdm(range(150)):
    if idx > limit > 0: break
    with open(f'MLD_data/{idx}.pickle', 'rb') as fh:
      data = pkl.load(fh)
      H = data['H']
      y = data['y']
      bits = data['bits']
      nbps: int = data['num_bits_per_symbol']
      SNR: int = data['SNR']

      H    = torch.from_numpy(H)   .to(device, torch.complex64)
      y    = torch.from_numpy(y)   .to(device, torch.complex64)
      bits = torch.from_numpy(bits).to(device, torch.float32)
      dataset.append([H, y, bits, nbps, SNR])
  return dataset

def compute_ber(solution: ndarray, bits: ndarray) -> float:
  '''
  Compute BER for the solution from QAIAs.

  Firstly, both the solution from QAIAs and generated bits should be transformed into gray-coded,
  and then compute the ber.

  Reference
  ---------
  [1] Kim M, Venturelli D, Jamieson K. Leveraging quantum annealing for large MIMO processing in centralized radio access networks.
      Proceedings of the ACM special interest group on data communication. 2019: 241-255.\

  Input
  -----
  solution: [rb*2*Nt, ], np.int
      The binary array filled with ones and minus ones.

  bits: [Nt, nbps], np.int
      The binary array filled with ones and zeros.
  Ouput
  -----
  ber: np.float
      A scalar, the BER.
  '''
  solution = solution.astype(np.int32)
  bits = bits.astype(np.int32)

  # convert the bits from sionna style to constellation style
  # Sionna QAM16 map: https://nvlabs.github.io/sionna/examples/Hello_World.html
  '''
  [sionna-style]
      1011 1001 0001 0011
      1010 1000 0000 0010
      1110 1100 0100 0110
      1111 1101 0101 0111
  [constellation-style] i.e. the "gray code" in QuAMax paper
      0010 0110 1110 1010
      0011 0111 1111 1011
      0001 0101 1101 1001
      0000 0100 1100 1000
  '''
  bits_constellation = 1 - np.concatenate([bits[..., 0::2], bits[..., 1::2]], axis=-1)

  # Fig. 2 from arXiv:2001.04014, the QuAMax paper converting QuAMax to gray coded
  num_bits_per_symbol = bits_constellation.shape[1]
  rb = num_bits_per_symbol // 2
  bits_hat = solution.reshape(rb, 2, -1)  # [rb, c=2, Nt]
  bits_hat = np.concatenate([bits_hat[:, 0], bits_hat[:, 1]], axis=0)  # [2*rb, Nt]
  bits_hat = bits_hat.T.copy()  # [Nt, 2*rb]
  bits_hat[bits_hat == -1] = 0  # convert Ising {-1, 1} to QUBO {0, 1}
  # QuAMax => intermediate code
  '''
  [QuAMax-style]
      0011 0111 1011 1111
      0010 0110 1010 1110
      0001 0101 1001 1101
      0000 0100 1000 1100
  [intermediate-style]
      0011 0100 1011 1100
      0010 0101 1010 1101
      0001 0110 1001 1110
      0000 0111 1000 1111
  '''
  output_bit = bits_hat.copy()  # copy b[0]
  index = np.nonzero(bits_hat[:, rb - 1] == 1)[0]  # select even columns
  bits_hat[index, rb:] = 1 - bits_hat[index, rb:]  # invert bits of high part (flip upside-down)
  # Differential bit encoding, intermediate code => gray code (constellation-style)
  for i in range(1, num_bits_per_symbol):  # b[i] = b[i] ^ b[i-1]
    output_bit[:, i] = np.logical_xor(bits_hat[:, i], bits_hat[:, i - 1])
  # calc BER
  ber = np.mean(bits_constellation != output_bit)
  return ber

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

      H, y, bits, nbps, SNR = sample
      if not args.overfit:
        bits, y = make_random_transmit(bits.shape, H, nbps, SNR)

      J, h = to_ising(H, y, nbps)
      spins = model(J, h, nbps)
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

      if not 'debug best pred':
        with torch.no_grad():
          soluts = torch.sign(spins).detach().cpu().numpy()
          bits_np = bits.cpu().numpy()
          ber = [compute_ber(solut, bits_np) for solut in soluts]
          print('ber:', ber)
          breakpoint()

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
    }
    print('params:', params)

    with open(LOG_PATH / f'{exp_name}.json', 'w', encoding='utf-8') as fh:
      json.dump(params, fh, indent=2, ensure_ascii=False)

  plt.plot(losses)
  plt.tight_layout()
  plt.savefig(LOG_PATH / f'{exp_name}.png', dpi=600)

def make_random_transmit(bits_shape:torch.Size, H:Tensor, nbps:int, SNR:int) -> Tuple[Tensor, Tensor]:
  # transmit random bits through given channel mix H
  bits = np.random.uniform(size=bits_shape) < 0.5
  x, y = modulate_and_transmit(bits.astype(np.float32), H.cpu().numpy(), nbps, SNR=10)   # SNR
  bits = torch.from_numpy(bits).to(device, torch.float32)
  y    = torch.from_numpy(y)   .to(device, torch.complex64)
  return bits, y


def ber_loss(spins:Tensor, bits:Tensor, loss_fn:str='mse') -> Tensor:
  ''' differentiable version of compute_ber() '''
  if False:
    from judger import compute_ber
    assert compute_ber

  # convert the bits from sionna style to constellation style
  # Sionna QAM16 map: https://nvlabs.github.io/sionna/examples/Hello_World.html
  bits_constellation = 1 - torch.cat([bits[..., 0::2], bits[..., 1::2]], dim=-1)

  # Fig. 2 from arXiv:2001.04014, the QuAMax paper converting QuAMax to gray coded
  nbps = bits_constellation.shape[1]
  rb = nbps // 2
  spins = torch.reshape(spins, (rb, 2, -1))  # [rb, c=2, Nt]
  spins = torch.permute(spins, (2, 1, 0))    # [Nt, c=2, rb]
  spins = torch.reshape(spins, (-1, 2*rb))   # [Nt, 2*rb]
  bits_hat = (spins + 1) / 2                 # Ising {-1, +1} to QUBO {0, 1}

  # QuAMax => intermediate code
  bits_final = bits_hat.clone()                           # copy b[0]
  index = torch.nonzero(bits_hat[:, rb-1] > 0.5)[:, -1]   # select even columns
  bits_hat[index, rb:] = 1 - bits_hat[index, rb:]         # invert bits of high part (flip upside-down)
  # Differential bit encoding, intermediate code => gray code (constellation-style)
  for i in range(1, nbps):                                # b[i] = b[i] ^ b[i-1]
    x = bits_hat[:, i] + bits_hat[:, i - 1]
    x_dual = 2 - x
    bits_final[:, i] = torch.where(x <= x_dual, x, x_dual)
  # calc BER
  if loss_fn in ['l2', 'mse']:
    return F.mse_loss(bits_final, bits_constellation)
  elif loss_fn in ['l1', 'mae']:
    return F.l1_loss(bits_final, bits_constellation)
  elif loss_fn == 'bce':
    pseudo_logits = bits_final * 2 - 1
    return F.binary_cross_entropy_with_logits(pseudo_logits, bits_constellation)

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
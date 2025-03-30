import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from numpy import ndarray

from qaia import QAIA, NMFA, SimCIM, CAC, CFC, SFC, ASB, BSB, DSB, LQA
from qaia import DUSB

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log'
DU_SB_weights = LOG_PATH / 'DU-SB_T=10_lr=0.0001.json'


run_cfg = 'DU_SB'
# run_cfg = 'baseline'
if run_cfg.startswith('DU_SB'):
    with open(DU_SB_weights, 'r', encoding='utf-8') as fh:
        params = json.load(fh)
        deltas: List[float] = params['deltas']
        eta: float = params['eta']
        a: float = params['a']

J_h = Tuple[ndarray, ndarray]

I_cache: Dict[int, ndarray] = {}
def get_I(N:int) -> ndarray:
    key = N
    if key not in I_cache:
        I_cache[key] = np.eye(N)
    return I_cache[key]

ones_cache: Dict[int, ndarray] = {}
def get_ones(N:int) -> ndarray:
    key = N
    if key not in ones_cache:
        ones_cache[key] = np.ones((N, 1))
    return ones_cache[key]

T_cache: Dict[Tuple[int, int], ndarray] = {}
def get_T(N:int, rb:int) -> ndarray:
    key = (N, rb)
    if key not in T_cache:
        # Eq. 7 the transform matrix T
        I = get_I(N)
        # [rb, N, N]
        T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
        # [rb*N, N] => [N, rb*N]
        T = T.reshape(-1, N).T
        T_cache[key] = T
    return T_cache[key]


def np_linagl_inv_hijack(a):
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)

    signature = 'D->D' if isComplexType(t) else 'd->d'
    extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    ainv = _umath_linalg.inv(a, signature=signature, extobj=extobj)
    return wrap(ainv.astype(result_t, copy=False))


def solver_qaia_lib(qaia_cls, J:ndarray, h:ndarray) -> ndarray:
    bs = 1
    solver: QAIA = qaia_cls(J, h, batch_size=bs, n_iter=10)
    solver.update()                     # [rb*N, B]
    if bs > 1:
        energy = min(solver.calc_energy())  # [1, B]
        opt_index = np.argmin(energy)
    else:
        opt_index = 0
        energy = min(solver.calc_energy())
    solution = np.sign(solver.x[:, opt_index])  # [rb*N], vset {-1, 1}
    return solution, energy

def solver_DU_SB(J:ndarray, h:ndarray) -> ndarray:
    global deltas, eta, a
    bs = 1
    solver = DUSB(J, h, deltas, eta, a, batch_size=bs)
    solver.update()                     # [rb*N, B]
    if bs > 1:
        energy = min(solver.calc_energy())   # [1, B]
        opt_index = np.argmin(energy)
    else:
        opt_index = 0
        energy = min(solver.calc_energy())
    solution = np.sign(solver.x[:, opt_index])  # [rb*N], vset {-1, 1}
    return solution, energy


def ising_generator(H:ndarray, y:ndarray, nbps:int, snr:float) -> J_h:
    if run_cfg == 'baseline':
        return to_ising(H, y, nbps)
    elif run_cfg == 'DU_SB':
        return to_ising(H, y, nbps)


def qaia_mld_solver(J:ndarray, h:ndarray) -> ndarray:
    if run_cfg == 'baseline':
        return solver_qaia_lib(BSB, J, h)
    elif run_cfg == 'DU_SB':
        return solver_DU_SB(J, h)

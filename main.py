import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from numpy import ndarray
import re
from qaia import QAIA, NMFA, SimCIM, CAC, CFC, SFC, ASB, BSB, DSB, LQA
from qaia import DUSB

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log'


J_h = Tuple[ndarray, ndarray]

def to_ising(H:ndarray, y:ndarray, nbps:int) -> J_h:
    '''
    Reduce MIMO detection problem into Ising problem.

    Reference
    ---------
    [1] Singh A K, Jamieson K, McMahon P L, et al. Ising machines’ dynamics and regularization for near-optimal mimo detection. 
        IEEE Transactions on Wireless Communications, 2022, 21(12): 11080-11094.
    [2] Ising Machines’ Dynamics and Regularization for Near-Optimal Large and Massive MIMO Detection. arXiv: 2105.10535v3

    Input
    -----
    H: [Nr, Nt], np.complex
        Channel matrix
    y: [Nr, 1], np.complex
        Received signal
    num_bits_per_symbol: int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    Output
    ------
    J: [rb*2*Nt, rb*2*Nt], np.float
        The coupling matrix of Ising problem
    h: [rb*2*Nt, 1], np.float
        The external field
    '''

    # the size of constellation, the M-QAM where M in {16, 64, 256}
    M = 2**nbps
    # n_elem at TX side (c=2 for real/imag, 1 symbol = 2 elem)
    Nr, Nt = H.shape
    N = 2 * Nt
    # n_bits/n_spins that one elem decodes to
    rb = nbps // 2

    # QAM variance for normalization
    # ref: https://dsplog.com/2007/09/23/scaling-factor-in-qam/
    #qam_var: float = 1 / (2**(rb - 2)) * np.sum(np.linspace(1, 2**rb - 1, 2**(rb - 1))**2)
    qam_var = 2 * (M - 1) / 3

    # Eq. 7 the transform matrix T
    I = np.eye(N)
    # [rb, N, N]
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    # [rb*N, N] => [N, rb*N]
    T = T.reshape(-1, N).T

    # Eq. 4 and 5
    H_tilde = np.vstack([
        np.hstack([H.real, -H.imag]), 
        np.hstack([H.imag,  H.real]),
    ])
    y_tilde = np.concatenate([y.real, y.imag])

    # Eq. 8, J is symmetric with diag=0, J[i,j] signifies spin interaction of σi and σj in the Ising model
    # This is different from the original paper because we use normalized transmitted symbol
    # J = -ZeroDiag(T.T * H.T * H * T))
    J = T.T @ H_tilde.T @ H_tilde @ T * (-2 / qam_var)
    J[np.diag_indices_from(J)] = 0
    # h = 2 * H * T.T * H.T * (y - H * T * 1 + (sqrt(M) - 1) * H * 1)
    z = y_tilde / np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1)) / qam_var + (np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1)) / qam_var
    h = 2 * z.T @ H_tilde @ T

    # [rb*N, rb*N], [rb*N, 1]
    return J, h.T


def solver_qaia_lib(qaia_cls, J:ndarray, h:ndarray, n_iter) -> ndarray:
    bs = 1
    solver: QAIA = qaia_cls(J, h, batch_size=bs, n_iter=n_iter)
    solver.update()                     # [rb*N, B]
    if bs > 1:
        energy = min(solver.calc_energy())  # [1, B]
        opt_index = np.argmin(energy)
    else:
        opt_index = 0
        energy = min(solver.calc_energy())
    solution = np.sign(solver.x[:, opt_index])  # [rb*N], vset {-1, 1}
    return solution, energy

def solver_DU_SB(J:ndarray, h:ndarray, deltas, eta, a) -> ndarray:
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
    return to_ising(H, y, nbps)

def qaia_mld_solver(J:ndarray, h:ndarray, run_cfg, DU_SB_weights) -> ndarray:
    # 获取文件名字符串，例如 "DU-SB_T=10_lr=0.0001.json"
    filename = DU_SB_weights.name
    pattern = r'T=(\d+)'
    match = re.search(pattern, filename)
    n_iter = int(match.group(1))
    if run_cfg == 'baseline':
        return solver_qaia_lib(BSB, J, h, n_iter)
    elif run_cfg == 'DU_SB':
        with open(DU_SB_weights, 'r', encoding='utf-8') as fh:
            params = json.load(fh)
            deltas: List[float] = params['deltas']
            eta: float = params['eta']
            a: float = params['a']
        return solver_DU_SB(J, h, deltas, eta, a)

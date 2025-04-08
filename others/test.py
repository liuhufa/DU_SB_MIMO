import pickle
from time import time
from glob import glob
import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path

# run_cfg='baseline'
run_cfg='DU_SB'
BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log'
DU_SB_weights = LOG_PATH / 'DU-SB_T=15_lr=0.0001.json'


class Judger:

    def __init__(self, test_cases):
        self.test_cases = test_cases

    @staticmethod
    def infer(J, h, qaia_mld_solver):
        bits = qaia_mld_solver(J, h, run_cfg, DU_SB_weights)
        return bits

    def benchmark(self, qaia_mld_solver):
        from collections import defaultdict
        energy_list = []
        obj_list = []

        avgenergy = 0
        avgobj = 0
        t1 = time()
        for i, case in enumerate(tqdm(self.test_cases)):
            J, h, bits, obj_val = case
            bits_decode, energy = self.infer(J, h, qaia_mld_solver)
            avgenergy += energy[0]
            avgobj += obj_val
            print(f'[case {i}] energy: {energy[0]}, obj: {obj_val}')

            energy_list.append(energy)
            obj_list.append(obj_val)
        t2 = time()
        avgobj /= len(self.test_cases)
        avgenergy /= len(self.test_cases)

        global run_cfg
        avgsbenergy = 0
        run_cfg = 'baseline'
        sbenergy_list = []
        t3 = time()
        for i, case in enumerate(tqdm(self.test_cases)):
            J, h, bits, obj_val = case
            bits_decode, energy = self.infer(J, h, qaia_mld_solver)
            avgsbenergy += energy[0]
            print(f'[case {i}] ber: {energy[0]}, ref_ber: {obj_val}')
            sbenergy_list.append(energy)
        t4 = time()
        avgsbenergy /= len(self.test_cases)

        if 'plot':
            from pathlib import Path
            BASE_PATH = Path(__file__).parent
            LOG_PATH = BASE_PATH / 'log';
            LOG_PATH.mkdir(exist_ok=True)
            pairs = list(zip(obj_list, energy_list, sbenergy_list))
            pairs.sort(reverse=True)  # decrease order by G_energy
            DUSB_list = [D_energy for G_energy, D_energy, SB_energy in pairs]
            SB_list = [SB_energy for G_energy, D_energy, SB_energy in pairs]
            Gurobi_list = [G_energy for G_energy, D_energy, SB_energy in pairs]
            plt.plot(DUSB_list, label=f'DUSB')
            plt.plot(SB_list, label=f'SB')
            plt.plot(Gurobi_list, label='Gurobi')
            plt.legend()
            plt.suptitle('Energy')
            plt.tight_layout()
            plt.savefig(LOG_PATH / 'solut.png', dpi=400)
            plt.show()
            plt.close()

        tt1 = t2 - t1
        tt2 = t4 - t3
        return avgenergy,avgsbenergy, avgobj, tt1, tt2


if __name__ == "__main__":
    from main import qaia_mld_solver

    dataset = []
    folder = 'test_dataset'
    files = sorted(os.listdir(folder))
    N = 12  # 固定变量数
    for idx, filename in enumerate(tqdm(files)):
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
            x = label.get("x", None)
        if label is not None:
            obj_val = label.get("obj_val", None)
        J = -J
        h = -h
        dataset.append((J, h, x, obj_val))

    judger = Judger(dataset)
    avgenergy, avgsbenergy, avgobj, tt1, tt2 = judger.benchmark(qaia_mld_solver)

    print(f'>> Method: DUSB')
    print(f'>> time cost: {tt1:.2f}')
    print(f">> avg. energy = {avgenergy:.5f}")
    print(f'>> Method: SB')
    print(f'>> time cost: {tt2:.2f}')
    print(f">> avg. energy = {avgsbenergy:.5f}")
    print(f">> avg. obj = {avgobj:.5f}")

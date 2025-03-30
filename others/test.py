import pickle
from time import time
from glob import glob
import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class Judger:

    def __init__(self, test_cases):
        self.test_cases = test_cases

    @staticmethod
    def infer(J, h, qaia_mld_solver):
        bits = qaia_mld_solver(J, h)
        return bits

    def benchmark(self, qaia_mld_solver):
        from collections import defaultdict
        energy_list = []
        obj_list = []

        avgenergy = 0
        avgobj = 0
        for i, case in enumerate(tqdm(self.test_cases)):
            J, h, bits, obj_val = case
            bits_decode, energy = self.infer(J, h, qaia_mld_solver)
            if i == 44:
                print(i)
            avgenergy += energy[0]
            avgobj += obj_val
            print(f'[case {i}] ber: {energy[0]}, ref_ber: {obj_val}')

            energy_list.append(energy)
            obj_list.append(obj_val)

        avgobj /= len(self.test_cases)
        avgenergy /= len(self.test_cases)
        return avgenergy, avgobj


if __name__ == "__main__":
    from main import ising_generator, qaia_mld_solver

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
    t = time()
    avgenergy, avgobj = judger.benchmark(qaia_mld_solver)
    ts = time() - t
    print(f'>> time cost: {ts:.2f}')
    print(f">> avg. energy = {avgenergy:.5f}")
    print(f">> avg. obj = {avgobj:.5f}")

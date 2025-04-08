from typing import List

import numpy as np
from numpy import ndarray

from .SB import BSB
import matplotlib.pyplot as plt
''' DU-SB in [arXiv:2306.16264] Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection '''

σ = lambda x: 1 / (1 + np.exp(-x))
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (sw(Λ * (x + 1)) - sw(Λ * (x - 1))) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (np.abs(x) - B))


# for inference
class DUSB(BSB):

    def __init__(self, J:ndarray, h:ndarray, deltas:List[float], eta:float, a:List[float], x:ndarray=None, batch_size:int=100):
        super().__init__(J, h, x, len(deltas), batch_size, dt=-1, xi=None)

        self.Δ = deltas
        self.η = eta

        self.c_0 = self.xi * self.η
        self.a_m1 = np.array(a) - 1

    def update(self):
        x_history = []
        for k, Δ_k in enumerate(self.Δ):
            self.y += Δ_k * (self.a_m1[k] * self.x + self.c_0 * (self.J @ self.x + self.h))
            self.x += Δ_k * self.y
            self.x = φ_s(self.x)
            self.y *= 1 - ψ_s(self.x)

            cond = np.abs(self.x) > 1
            x_record = np.where(cond, np.sign(self.x), self.x)  # limit x to vrng [-1, +1]
            x_history.append(x_record)

        """分岔图绘制函数"""
        # self.plot_bifurcation(x_history)
    def plot_bifurcation(self, x_history):
        """绘制分岔图：展示 self.x 随迭代的变化情况"""
        plt.figure(figsize=(10, 6))

        # 绘制每个变量（x 向量的每个元素）随迭代的变化
        num_vars = self.x.shape[0]  # 变量的数量，即 x 向量的长度
        colors = plt.cm.jet(np.linspace(0, 1, num_vars))  # 使用 jet 色图为每个变量选择不同颜色

        for j in range(num_vars):
            # 绘制每个变量的迭代过程
            x_var_history = [x_iter[j] for x_iter in x_history]  # 获取第 j 个变量的历史
            plt.plot(range(self.n_iter), x_var_history, label=f"Variable {j + 1}", color=colors[j])

        plt.title("Bifurcation of each variable over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("x value")
        plt.legend()
        plt.grid(True)
        plt.show()
    def update_hard(self):
        for k, Δ_k in enumerate(self.Δ):
            self.y += Δ_k * (self.a_m1[k] * self.x + self.c_0 * (self.J @ self.x + self.h))
            self.x += Δ_k * self.y

            cond = np.abs(self.x) > 1
            self.x = np.where(cond, np.sign(self.x), self.x)          # limit x to vrng [-1, +1]
            self.y = np.where(cond, np.zeros_like(self.x), self.y)    # if |x|==1 is fully annealled, set y to zero


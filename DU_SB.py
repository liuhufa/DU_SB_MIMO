# Author: PAFF
# CreatTime: 3/17/2025
# FileName: Deepunfolding SB
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.storage


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


σ = F.sigmoid
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (sw(Λ * (x + 1)) - sw(Λ * (x - 1))) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (torch.abs(x) - B))

# 基本的DU_SB
class DU_SB(nn.Module):

  # arXiv:2306.16264 Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection
  # 构造函数，初始化模型的主要参数
  # T 是迭代次数
  # batch_size 是每批次处理的样本数量

  def __init__(self, T:int, batch_size:int=100):
    super().__init__()

    self.T = T
    self.batch_size = batch_size

    # 退火值a 初始值为在0到1之间生成线性等间隔张量
    self.a = Parameter(torch.linspace(0, 1, T), requires_grad=True)

    # 长度为T的一维张量，初始值为1
    self.Δ = Parameter(torch.ones  ([T],    dtype=torch.float32), requires_grad=True)
    # 标量张量，初始值1
    self.η = Parameter(torch.tensor([1.0],  dtype=torch.float32), requires_grad=True)


  def forward(self, J:Tensor, h:Tensor, **kwargs) -> Tensor:

    B = self.batch_size
    N = J.shape[0]
    c_0: Tensor = 0.5 * math.sqrt(N - 1) / torch.linalg.norm(J, ord='fro')

    # rand init x and y
    x = 0.02 * (torch.rand(N, B, device=J.device) - 0.5)
    y = 0.02 * (torch.rand(N, B, device=J.device) - 0.5)

    # 进行一次bSB
    for k, Δ_k in enumerate(self.Δ):
      y = y + Δ_k * (-(1 - self.a[k]) * x + self.η * c_0 * (J @ x + h))
      x = x + Δ_k * y
      x = φ_s(x)
      y = y * (1 - ψ_s(x))

    spins = x.T

    return spins


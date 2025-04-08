# 深度展开量子启发式算法

## 1. 使用说明

1. `DU_SB.py`是深度展开SB算法的训练模型，可训练参数为`a`,`Δ`和`η`
2. `train.py`是以MIMO问题为例的训练过程，训练集为根据H矩阵和信噪比自动生成，训练后的参数存在`/log/.json`
3. `test.py`是测试训练结果，与ZF方法的对比结果存在`/log/solut.png`
4. `main.py`存有测试过程中的参数读取和测试方法选择，可选方法为`DU_SB`即深度展开SB，和`baseline`即BSB基线方法
5. `/qaia/DUSB.py`包含深度展开的测试时调用的函数，需要首先创建DUSB类（输入ising模型参数`J,h`,先前训练好的参数`Δ,η`以及`batch_size`）

## 2. 测试效果

1. `bs=1, T=10`的情况下，DU_SB的平均误码率为23.70%，SB的平均误码率为28.48%
2. 根据图`/log/solut.png`，DU_SB的结果在大多数案例上误码率都远小于ZF方法

## 3. Others

1. `others`文件夹中是一个测试案例，dataset中为随机生成的数据集，决策变量为`n=12`个，随机生成了变量之间的相关系数。要求最小化哈密顿量`H = sum(0.5 * J[i][j] * z[i] * z[j] for i in range(n) for j in range(n)) + sum(h[i] * z[i] for i in range(n))`
2. `dataset`中的`label`由商业求解器Gurobi计算得到。
3. `bs=1, T=15`的情况下，DU_SB的平均能量值为-85.13(gap:5.18%)，SB的平均能量值为-82.80(gap:7.77%)，Gurobi的平均能量值为-89.7796。

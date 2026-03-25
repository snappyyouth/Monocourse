https://fudan-nlp.feishu.cn/wiki/VhCHwZiXQicIYrkbd2ocFS6Kn9c

# PyTorch 内置神经网络函数 vs 手动实现对比分析

本项目 `main.py` 完全使用 PyTorch 基础张量运算实现了一个文本情感分类器，
不依赖 `torch.nn`、`torch.optim`、`torch.utils.data` 等高层模块。
以下从各个环节逐一对比两者的差异。

---

## 1. 模型定义

### PyTorch 内置方式

```python
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(vocab_size, num_classes)

    def forward(self, x):
        return self.fc(x)
```

- 继承 `nn.Module`，参数自动注册、自动追踪
- `nn.Linear` 内部自动创建 `weight` 和 `bias`，并绑定到计算图

### 手动实现（main.py）

```python
std = (2.0 / (vocab_size + num_classes)) ** 0.5
W = torch.randn(vocab_size, num_classes) * std   # 手动 Xavier 初始化
b = torch.zeros(num_classes)
```

- 参数就是普通张量，需要自行管理
- 初始化策略需要自己计算（此处手写 Xavier 公式）

### 核心差异

| 方面 | 内置 `nn.Module` | 手动张量 |
|------|-------------------|----------|
| 参数管理 | `model.parameters()` 自动收集 | 需手动跟踪每个 W、b |
| 初始化 | `nn.Linear` 默认 Kaiming 初始化 | 需自行实现初始化公式 |
| 设备迁移 | `model.to(device)` 一键迁移 | 每个张量单独 `.to(device)` |
| 序列化 | `model.state_dict()` 标准接口 | 自行组织 dict 存储 |

---

## 2. 前向传播

### PyTorch 内置方式

```python
logits = model(x)  # 自动调用 forward()，经过 autograd 计算图
```

- `nn.Linear` 内部执行 `F.linear(x, weight, bias)`，即 `x @ weight.T + bias`
- 每步运算自动构建计算图，为反向传播做准备

### 手动实现（main.py）

```python
def forward(X, W, b):
    logits = X @ W + b
    probs = softmax(logits)
    return logits, probs
```

- 直接用 `@`（矩阵乘法）和 `+`（广播加法）
- 没有计算图，不支持自动求导

### 核心差异

内置前向传播的每一步张量运算都会在后台记录操作历史（计算图），
手动实现则只是纯粹的数值计算，不保留任何梯度信息。

---

## 3. 损失函数

### PyTorch 内置方式

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)  # 内部: log_softmax + nll_loss
```

- 将 softmax 和 log 合并为 `log_softmax`，用 LogSumExp 技巧保证数值稳定
- 直接接受原始 logits，无需先算 softmax

### 手动实现（main.py）

```python
def softmax(logits):
    max_val = logits.max(dim=1, keepdim=True).values
    exp = torch.exp(logits - max_val)          # 减最大值防溢出
    return exp / exp.sum(dim=1, keepdim=True)

def cross_entropy_loss(probs, labels):
    correct_probs = probs[torch.arange(n), labels]
    return -torch.log(correct_probs + 1e-12).mean()  # 加 epsilon 防 log(0)
```

- 分两步：先 softmax 再取 -log
- 需要自己处理数值稳定性（减最大值、加 epsilon）

### 核心差异

| 方面 | 内置 `CrossEntropyLoss` | 手动实现 |
|------|-------------------------|----------|
| 数值稳定性 | LogSumExp 一步到位，精度最优 | 分两步走，需手动处理溢出/下溢 |
| 输入 | 直接接受 raw logits | 需要先算 softmax 概率 |
| 实现复杂度 | 一行调用 | 两个函数约 10 行 |

---

## 4. 反向传播

### PyTorch 内置方式

```python
loss.backward()  # autograd 自动沿计算图反向求梯度
# 梯度存在 param.grad 中
```

- 自动微分引擎遍历计算图，对每个算子执行链式法则
- 支持任意复杂的网络结构，无需手写梯度

### 手动实现（main.py）

```python
def backward(X, probs, labels, W):
    one_hot = torch.zeros(n, num_classes)
    one_hot[torch.arange(n), labels] = 1.0
    d_logits = (probs - one_hot) / n       # softmax+CE 的联合梯度

    grad_W = X.t() @ d_logits               # 链式法则: dL/dW = X^T · dL/dz
    grad_b = d_logits.sum(dim=0)            # dL/db = Σ dL/dz
    return grad_W, grad_b
```

- 需要自己推导 softmax + cross_entropy 的梯度公式
- 对于 softmax + CE 的特殊情况，梯度恰好是 `probs - one_hot`，公式简洁
- 但如果网络更深（多层、非线性激活），手动推导和实现的复杂度会急剧增长

### 核心差异

| 方面 | 内置 autograd | 手动求导 |
|------|---------------|----------|
| 通用性 | 任意计算图自动求导 | 每换一种网络结构就要重新推导 |
| 正确性 | 经过大量验证，极少出错 | 容易因公式推导错误导致训练异常 |
| 扩展难度 | 加层只需改 forward | 每加一层就要新增一段梯度代码 |
| 教学价值 | 黑盒，不直观 | 直接理解梯度流动过程 |

---

## 5. 优化器

### PyTorch 内置方式

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 训练循环中:
optimizer.zero_grad()
loss.backward()
optimizer.step()   # 自动更新所有参数，含动量/自适应学习率
```

- 提供 SGD、Adam、AdamW 等多种优化算法
- 自动管理梯度清零、动量缓存、学习率衰减等

### 手动实现（main.py）

```python
W = W - lr * grad_W
b = b - lr * grad_b
```

- 只实现了最基本的 SGD：`θ = θ - lr * ∇θ`
- 如需 Adam 等高级优化器，要手动维护一阶矩 m、二阶矩 v、偏差修正等状态

### 核心差异

手动实现 SGD 只需一行，但 Adam 需要额外维护约 6 个状态变量和多步更新逻辑，
实现工作量和出错概率显著增加。

---

## 6. 数据加载

### PyTorch 内置方式

```python
from torch.utils.data import Dataset, DataLoader

dataset = MyDataset(file_path)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
for batch_x, batch_y in loader:
    ...
```

- `DataLoader` 自动处理 batching、shuffling、多进程并行加载
- `Dataset` 提供统一的 `__getitem__` / `__len__` 接口

### 手动实现（main.py）

```python
perm = torch.randperm(n_train)
X_train = X_train[perm]
y_train = y_train[perm]

for start in range(0, n_train, batch_size):
    end = min(start + batch_size, n_train)
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]
```

- 用 `torch.randperm` 打乱索引，手动切片取 mini-batch
- 无并行加载，所有数据需预先加载到内存

### 核心差异

| 方面 | 内置 DataLoader | 手动切片 |
|------|-----------------|----------|
| 并行加载 | `num_workers` 多进程预取 | 无，单线程顺序读取 |
| 内存管理 | 支持惰性加载（按需读样本） | 所有数据必须一次性放入内存 |
| Shuffle | 内置、每个 epoch 自动打乱 | 需手动 `randperm` + 索引重排 |

---

## 7. 总结

```
                内置 PyTorch                          手动实现
              ┌─────────────┐                   ┌──────────────────┐
  模型定义    │ nn.Module    │                   │ 裸张量 W, b       │
              │ nn.Linear    │                   │ 手动初始化        │
              ├─────────────┤                   ├──────────────────┤
  前向传播    │ 自动计算图    │                   │ X @ W + b         │
              ├─────────────┤                   ├──────────────────┤
  损失函数    │ CrossEntropy │                   │ 手写 softmax + CE │
              │ Loss         │                   │                  │
              ├─────────────┤                   ├──────────────────┤
  反向传播    │ loss.backward│                   │ 手推梯度公式       │
              │ (autograd)   │                   │ grad = X^T · dz  │
              ├─────────────┤                   ├──────────────────┤
  优化器      │ Adam/SGD     │                   │ W -= lr * grad_W  │
              ├─────────────┤                   ├──────────────────┤
  数据加载    │ DataLoader   │                   │ randperm + 切片   │
              └─────────────┘                   └──────────────────┘
```

**手动实现的价值**：深入理解前向传播、梯度计算、参数更新的底层原理，
这些正是 PyTorch 高层 API 在背后自动完成的事情。

**内置函数的价值**：工程效率、数值稳定性、通用性和可扩展性。
在实际项目中，应该使用 `torch.nn` 等高层接口，把精力放在模型设计和实验上。

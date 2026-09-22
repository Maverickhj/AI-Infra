# 卷积与 Conv3d：数学原理、几何直觉与 PyTorch 实现

## 0. 先建立卷积的几何心智模型

在进入卷积的具体数学公式之前，先建立一个更直观的理解：

> **卷积不是首先在做“滑动窗口求和”，而是在做局部模式匹配。**

对于某个局部输入 patch，将它展平成向量：

$$
\mathbf{x}\in\mathbb{R}^{K}
$$

将卷积核也展平成向量：

$$
\mathbf{w}\in\mathbb{R}^{K}
$$

那么某个位置上的卷积输出，本质上就是：

$$
y=\mathbf{w}^{T}\mathbf{x}
$$

也就是两个向量之间的内积。

从几何上：

$$
\mathbf{w}^{T}\mathbf{x}
=
\|\mathbf{w}\|
\|\mathbf{x}\|
\cos\theta
$$

其中 $\theta$ 表示局部 patch $\mathbf{x}$ 与卷积核 $\mathbf{w}$ 之间的夹角。

因此：

- 如果局部 patch 与 kernel 表示的模式很相似，卷积响应通常较大；
- 如果两者相关性较弱，内积接近 0；
- 如果两者方向相反，响应可能为负。

因此可以建立第一个核心心智模型：

$$
\boxed{
\text{卷积输出}
\approx
\text{局部 patch 与 kernel 模板的匹配程度}
}
$$

### 0.1 Kernel 可以理解为一种局部模式

例如：

$$
W=
\begin{bmatrix}
-1 & 0 & 1\\
-1 & 0 & 1\\
-1 & 0 & 1
\end{bmatrix}
$$

它描述的是一种局部模式：左侧值较小、右侧值较大。

如果某个输入 patch 具有类似结构：

$$
X=
\begin{bmatrix}
0 & 0 & 1\\
0 & 0 & 1\\
0 & 0 & 1
\end{bmatrix}
$$

那么 kernel 与 patch 会产生较大的内积。

因此，这个 kernel 可以理解为一个检测垂直亮度边界的局部模板。神经网络中的 kernel 参数则是通过训练学习得到的。

---

### 0.2 卷积也可以理解为局部特征投影

从线性代数角度，kernel $\mathbf{w}$ 可以看成局部特征空间中的一个方向。

输入 patch $\mathbf{x}$ 与它做内积：

$$
\mathbf{w}^{T}\mathbf{x}
$$

就是在测量：

> 输入 patch 在 $\mathbf{w}$ 这个特征方向上有多大的分量。

因此还可以建立第二个心智模型：

$$
\boxed{
\text{Convolution}
=
\text{在每个局部位置，将输入投影到某个特征方向}
}
$$

如果有多个 output channel，也就是有多个 kernel：

$$
\mathbf{w}_1,\mathbf{w}_2,\dots,\mathbf{w}_{C_{\text{out}}}
$$

那么同一个局部 patch 会得到多个不同模式上的响应。

---

### 0.3 Feature Map 的几何含义

如果某个 kernel 学会检测“垂直边缘”，那么它在整张图片上滑动时，每个位置都会得到一个响应值：

$$
y[h,w]
$$

最终形成一张 feature map。

因此：

$$
\boxed{
\text{Feature Map}
=
\text{某一种局部模式在空间中的响应分布}
}
$$

也就是说，feature map 可以理解为：

> 某个 learned pattern 在哪些位置出现，以及出现得有多强。

---

### 0.4 Weight Sharing 的几何意义

同一个 kernel 会在所有空间位置重复使用。

因此如果一个 kernel 学会检测某种模式，那么无论这个模式出现在图片左上角、中间还是右下角，都使用同一组参数进行检测。

所以 weight sharing 可以理解为：

> 同一种局部模式，不应该因为它移动到了不同位置，就重新学习一套检测规则。

这也带来了卷积的重要性质：translation equivariance。

---

### 0.5 Conv2d 与 Conv3d 的统一心智模型

Conv2d 的 kernel 观察：

$$
K_H\times K_W
$$

大小的局部区域，因此主要是在寻找二维空间中的局部 pattern，例如：

- 边缘；
- 纹理；
- 局部形状。

Conv3d 的 kernel 则观察：

$$
K_D\times K_H\times K_W
$$

大小的局部 volume。

如果 $D=T$ 表示时间，那么它是在寻找局部时空模式，例如：

- 向左或向右移动；
- 扩张；
- 收缩；
- 闪烁；
- 某种连续动作变化。

因此：

$$
\boxed{
\text{Conv2d}
=
\text{局部空间模式检测}
}
$$

$$
\boxed{
\text{Conv3d}
=
\text{局部时空模式检测}
}
$$

最后可以把整套心智模型压缩为：

$$
\boxed{
\text{Convolution}
=
\text{Local Pattern Matching}
+
\text{Weight Sharing}
}
$$

或者从线性代数角度：

$$
\boxed{
\text{Convolution}
=
\text{Local Feature Projection}
}
$$

---

## TODO

- [ ] 观看 **3Blue1Brown 的 convolution 介绍视频**，补充对卷积几何意义与直觉的理解。

---

## 1. 从数学上理解卷积

先不考虑神经网络。

对于一维连续函数 $x(t)$ 和卷积核 $k(t)$，数学上的卷积定义为：

$$
y(t)
=
(x * k)(t)
=
\int_{-\infty}^{+\infty}
x(\tau) k(t-\tau)\,d\tau
$$

它表达的事情可以理解为：

> 在位置 $t$，拿一个局部模板 $k$，和输入 $x$ 周围的一段区域做加权求和。

离散情况下：

$$
y[i]
=
\sum_u
x[i-u]k[u]
$$

可以把 $k$ 看成一个滑动的小窗口。

例如：

$$
k =
\begin{bmatrix}
-1 & 0 & 1
\end{bmatrix}
$$

在一维信号上滑动时，它实际上在比较：

$$
x[i+1] - x[i-1]
$$

因此它可以检测信号变化。

---

## 2. 神经网络里的卷积

PyTorch 中所谓的 convolution，严格来说通常实现的是 **cross-correlation（互相关）**，不会把 kernel 翻转。

因此更适合用下面这个公式描述神经网络中的卷积：

$$
y[i]
=
\sum_u
x[i+u]w[u]
$$

其中：

- $x$：输入；
- $w$：卷积核参数；
- $i$：当前位置；
- $u$：卷积核内部位置；
- $y$：输出。

神经网络通过训练学习 $w$。

因此可以把卷积理解为：

$$
\boxed{
\text{局部区域}
\rightarrow
\text{共享的线性变换}
}
$$

这是理解 CNN 最重要的核心之一。

---

## 3. 卷积解决了什么问题

假设输入是一张：

$$
224 \times 224
$$

的图片。

如果直接使用全连接层，那么每个输出神经元都连接：

$$
224 \times 224 = 50176
$$

个输入。

如果有 1000 个输出神经元，那么参数量约为：

$$
50176 \times 1000
\approx 5\times10^7
$$

图片本身具有两个非常重要的性质：**局部性**和**参数共享**。

### 3.1 局部性 Locality

一个像素附近的像素通常与它更加相关。

例如识别一条边，只需要观察附近几个像素，没有必要让左上角像素与右下角像素直接建立一套独立参数。

因此卷积只处理一个局部窗口，例如：

$$
3\times3
$$

对应：

$$
y[i,j]
=
\sum_{u=0}^{2}
\sum_{v=0}^{2}
w[u,v]
x[i+u,j+v]
$$

### 3.2 参数共享 Weight Sharing

更加关键的是，同一个卷积核 $w$ 会在整张图片上重复使用。

例如一个：

$$
3\times3
$$

kernel：

$$
W =
\begin{bmatrix}
w_{00} & w_{01} & w_{02}\\
w_{10} & w_{11} & w_{12}\\
w_{20} & w_{21} & w_{22}
\end{bmatrix}
$$

它会在图片的不同位置滑动，但所有位置使用的都是同一组 $W$。

因此卷积的核心优势可以概括为：

$$
\boxed{
\text{Locality}
+
\text{Weight Sharing}
}
$$

这使得参数量不再主要依赖整个输入尺寸，而主要与以下量相关：

$$
C_{\text{in}},
C_{\text{out}},
K
$$

---

## 4. Conv2d 的数学形式

先建立统一符号。

假设输入：

$$
X
\in
\mathbb{R}^{C_{\text{in}}\times H\times W}
$$

卷积核：

$$
W
\in
\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times K_H\times K_W}
$$

输出：

$$
Y
\in
\mathbb{R}^{C_{\text{out}}\times H_{\text{out}}\times W_{\text{out}}}
$$

那么忽略 stride 和 padding 时：

$$
Y[c_o,h,w]
=
b[c_o]
+
\sum_{c_i=0}^{C_{\text{in}}-1}
\sum_{i=0}^{K_H-1}
\sum_{j=0}^{K_W-1}
W[c_o,c_i,i,j]
X[c_i,h+i,w+j]
$$

对于每一个输出位置，都会取一个：

$$
C_{\text{in}}
\times K_H
\times K_W
$$

大小的局部区域。

然后与对应的 kernel 做一次内积。

---

## 5. Conv3d 是什么

Conv3d 并没有引入新的核心数学思想。

它只是将二维空间：

$$
H\times W
$$

扩展为三维空间：

$$
D\times H\times W
$$

因此输入：

$$
X
\in
\mathbb{R}^{
C_{\text{in}}
\times D
\times H
\times W
}
$$

卷积核：

$$
W
\in
\mathbb{R}^{
C_{\text{out}}
\times C_{\text{in}}
\times K_D
\times K_H
\times K_W
}
$$

这里统一使用：

- $D$：Depth；
- $H$：Height；
- $W$：Width。

Conv3d 的核心计算为：

$$
\boxed{
Y[c_o,d,h,w]
=
b[c_o]
+
\sum_{c_i}
\sum_i
\sum_j
\sum_k
W[c_o,c_i,i,j,k]
X[c_i,d+i,h+j,w+k]
}
$$

因此 Conv3d 本质上是在：

> $D,H,W$ 三个维度上同时滑动一个三维局部窗口。

---

## 6. Conv3d 在“看”什么

假设：

$$
K_D=3,\quad
K_H=3,\quad
K_W=3
$$

那么一个 kernel 会观察：

$$
3\times3\times3
$$

大小的局部区域，也就是一个小立方体。

对于视频来说，可以令：

$$
D=T
$$

其中 $T$ 表示时间。

输入可以写成：

$$
X
\in
\mathbb{R}^{C\times T\times H\times W}
$$

于是：

$$
K_D\times K_H\times K_W
$$

也可以写成：

$$
K_T\times K_H\times K_W
$$

例如：

$$
3\times3\times3
$$

表示同时观察连续 3 帧，并在每一帧中观察一个 $3\times3$ 的空间区域。

---

## 7. Conv3d 解决什么问题

二维卷积主要学习：

$$
(H,W)
$$

中的局部关系，例如：

- 边缘；
- 纹理；
- 局部形状。

但是对于视频，数据还多了时间维度：

$$
(T,H,W)
$$

如果一个物体在连续帧中移动，仅仅独立观察每一帧，只能知道“这一帧这里有一个物体”。

联合观察多个时间步，才能捕获：

$$
(t,h,w)
$$

之间的变化关系，例如物体的移动方向。

因此 Conv3d 可以同时学习：

$$
\boxed{
\text{spatial feature}
+
\text{temporal feature}
}
$$

即：

$$
\text{空间模式} + \text{时间变化}
$$

---

## 8. Conv1d、Conv2d、Conv3d 的统一理解

Conv2d 的局部感受野是：

$$
K_H \times K_W
$$

Conv3d 的局部感受野是：

$$
K_D \times K_H \times K_W
$$

因此可以统一理解为：

$$
1D
\rightarrow
2D
\rightarrow
3D
$$

只是卷积核滑动的空间维度不同。

核心数学思想并没有改变。

---

## 9. PyTorch 中 Conv3d 的输入形式

PyTorch 提供：

```python
torch.nn.Conv3d(
    in_channels,
    out_channels,
    kernel_size,
)
```

输入 shape 为：

$$
\boxed{
[N,C_{\text{in}},D,H,W]
}
$$

其中：

- $N$：batch size；
- $C_{\text{in}}$：输入 channel；
- $D$：depth，视频中通常可以理解为 time；
- $H$：height；
- $W$：width。

例如一个视频输入：

```python
x = torch.randn(
    8,      # batch
    3,      # RGB
    16,     # frames
    224,    # height
    224,    # width
)
```

其 shape 为：

$$
[8,3,16,224,224]
$$

---

## 10. 一个最简单的 PyTorch Conv3d

```python
import torch
import torch.nn as nn

conv = nn.Conv3d(
    in_channels=3,
    out_channels=64,
    kernel_size=(3, 3, 3),
)

x = torch.randn(8, 3, 16, 224, 224)

y = conv(x)

print(y.shape)
```

此时 kernel 参数 shape 为：

$$
[64,3,3,3,3]
$$

即：

$$
[
C_{\text{out}},
C_{\text{in}},
K_D,
K_H,
K_W
]
$$

对应：

```python
conv.weight.shape
# torch.Size([64, 3, 3, 3, 3])
```

---

## 11. 一个输出元素是如何计算的

假设：

```python
conv = nn.Conv3d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
)
```

某一个输出元素：

$$
Y[n,c_o,d,h,w]
$$

会读取输入中的局部区域：

$$
X[
n,
:,
d:d+3,
h:h+3,
w:w+3
]
$$

其 shape 为：

$$
[3,3,3,3]
$$

即：

$$
C_{\text{in}}
\times
K_D
\times
K_H
\times
K_W
$$

一共包含：

$$
3\times3\times3\times3
=
81
$$

个元素。

对应的 kernel：

$$
W[c_o]
$$

shape 也是：

$$
[3,3,3,3]
$$

然后执行 element-wise multiplication，再求和：

$$
Y[n,c_o,d,h,w]
=
\sum
X_{\text{local}}
\odot
W[c_o]
+
b[c_o]
$$

因此：

$$
\boxed{
\text{一个 output element}
=
\text{一个 local patch 与 kernel 的 dot product}
}
$$

---

## 12. Stride 和 Padding 如何进入公式

更一般的 Conv3d 公式可以写为：

$$
Y[c_o,d,h,w]
=
b[c_o]
+
\sum_{c_i,k_d,k_h,k_w}
W[c_o,c_i,k_d,k_h,k_w]
X[
c_i,
dS_D+k_d-P_D,
hS_H+k_h-P_H,
wS_W+k_w-P_W
]
$$

其中：

- $S_D,S_H,S_W$：stride；
- $P_D,P_H,P_W$：padding。

stride 决定 kernel 每次移动多少格。

例如：

$$
S_D=2
$$

表示卷积核在 depth 维度上每次移动两个位置。

---

## 13. 输出 Shape 如何计算

对于每个维度，都可以使用：

$$
L_{\text{out}}
=
\left\lfloor
\frac{
L_{\text{in}}
+
2P
-
D_{\text{ilation}}(K-1)
-
1
}{
S
}
+1
\right\rfloor
$$

其中：

- $L_{\text{in}}$：输入长度；
- $L_{\text{out}}$：输出长度；
- $P$：padding；
- $K$：kernel size；
- $S$：stride；
- $D_{\text{ilation}}$：dilation。

因此 Conv3d 分别计算：

$$
D_{\text{out}},
H_{\text{out}},
W_{\text{out}}
$$

例如：

$$
K=3,\quad
P=0,\quad
S=1
$$

则：

$$
D_{\text{out}}
=
D_{\text{in}}-2
$$

$$
H_{\text{out}}
=
H_{\text{in}}-2
$$

$$
W_{\text{out}}
=
W_{\text{in}}-2
$$

所以：

$$
[8,3,16,224,224]
$$

经过 $3\times3\times3$ Conv3d 后得到：

$$
[8,64,14,222,222]
$$

---

## 14. PyTorch 底层如何实现 Conv3d

这里要区分两件事：

1. 数学上的 Conv3d；
2. GPU 上真正执行的高性能 Conv3d。

概念上，Conv3d 可以写成多重循环：

```python
for n in range(N):
    for co in range(C_out):
        for d in range(D_out):
            for h in range(H_out):
                for w in range(W_out):

                    y[n, co, d, h, w] = 0

                    for ci in range(C_in):
                        for kd in range(KD):
                            for kh in range(KH):
                                for kw in range(KW):

                                y[n, co, d, h, w] += (
                                    x[n, ci, d + kd, h + kh, w + kw]
                                    * weight[co, ci, kd, kh, kw]
                                )
```

数学上这就是 Conv3d。

但是 GPU 实现通常不会真的执行 Python 风格的多层 for-loop。

---

## 15. Conv3d 与矩阵乘法的关系

Conv3d 的计算本质上包含大量：

$$
a\times b+c
$$

也就是 multiply-accumulate。

而 GPU 非常擅长执行：

$$
C = AB
$$

这种大规模矩阵乘法。

因此可以把一个局部 patch：

$$
C_{\text{in}}
\times
K_D
\times
K_H
\times
K_W
$$

展平为：

$$
\mathbf{x}_{\text{patch}}
\in
\mathbb{R}^{K}
$$

其中：

$$
K
=
C_{\text{in}}
K_DK_HK_W
$$

卷积核也展平为：

$$
\mathbf{w}
\in
\mathbb{R}^{K}
$$

于是一个输出值可以写为：

$$
y
=
\mathbf{x}_{\text{patch}}
\mathbf{w}
$$

如果把很多 patch 放在一起，就可以组织为：

$$
Y
=
X_{\text{patch}}W
$$

从而转化为大规模矩阵乘法。

---

## 16. im2col 思想

经典卷积实现中有一个著名思想：

$$
\boxed{\text{im2col}}
$$

对于 Conv3d，可以理解成把很多三维 local patch 展开成矩阵。

每一个 patch 展开后的长度是：

$$
C_{\text{in}}
K_DK_HK_W
$$

于是：

$$
X_{\text{patch}}
=
\begin{bmatrix}
\text{patch}_1\\
\text{patch}_2\\
\text{patch}_3\\
\vdots
\end{bmatrix}
$$

kernel 则可以整理为：

$$
W
\in
\mathbb{R}^{
C_{\text{out}}
\times
(C_{\text{in}}K_DK_HK_W)
}
$$

这样卷积计算就可以转换为矩阵乘法。

---

## 17. 现代 PyTorch 并不等于显式 im2col

调用：

```python
y = torch.nn.functional.conv3d(x, weight)
```

并不意味着 PyTorch 一定会执行：

```text
unfold
↓
生成完整 patch matrix
↓
torch.matmul
```

真实实现通常更接近：

```text
PyTorch
  ↓
ATen
  ↓
CUDA convolution backend
  ↓
cuDNN / specialized kernel
  ↓
GPU
```

根据以下因素：

- tensor shape；
- dtype；
- GPU 架构；
- kernel size；
- stride；
- padding；
- groups；
- memory layout；

底层可能选择不同的 convolution algorithm。

现代 GPU kernel 往往会在计算过程中直接完成：

- 数据加载；
- patch 访问；
- multiply-accumulate；
- 输出写回；

而不是先物化一个巨大的 im2col 中间 tensor。

---

## 18. 从 PyTorch 调用链理解 Conv3d

从 PyTorch 实现角度，可以粗略理解为：

```text
nn.Conv3d
   │
   │ forward()
   ▼
F.conv3d(...)
   │
   ▼
ATen convolution
   │
   ├── shape / stride / padding / groups
   │
   └── backend dispatch
           │
           ▼
        CUDA / cuDNN
           │
           ▼
      convolution kernel
```

`nn.Conv3d` 本身主要负责保存参数和配置，例如：

```python
self.weight
self.bias
self.stride
self.padding
self.dilation
self.groups
```

然后在 forward 中调用类似：

```python
F.conv3d(
    input,
    self.weight,
    self.bias,
    self.stride,
    self.padding,
    self.dilation,
    self.groups,
)
```

因此可以理解为：

$$
\boxed{
\texttt{nn.Conv3d}
=
\text{parameters}
+
\text{conv3d operator}
}
$$

---

## 19. 总结

整个 Conv3d 的认知链可以概括为：

```text
输入数据
X[N, C_in, D, H, W]
        │
        ▼
取局部 3D patch
[C_in, K_D, K_H, K_W]
        │
        ▼
与一个 kernel 做 inner product
        │
        ▼
得到一个输出值
Y[n, c_out, d, h, w]
        │
        ▼
kernel 在 D/H/W 上滑动
        │
        ▼
得到一个 feature map
        │
        ▼
使用 C_out 个不同 kernel
        │
        ▼
Y[N, C_out, D_out, H_out, W_out]
```

对应的核心公式为：

$$
\boxed{
Y[n,c_o,d,h,w]
=
b[c_o]
+
\sum_{c_i}
\sum_{k_d}
\sum_{k_h}
\sum_{k_w}
W[c_o,c_i,k_d,k_h,k_w]
X[
n,
c_i,
dS_D+k_d-P_D,
hS_H+k_h-P_H,
wS_W+k_w-P_W
]
}
$$

卷积背后的核心设计思想是：

$$
\boxed{
\text{Locality}
}
$$

$$
\boxed{
\text{Weight Sharing}
}
$$

$$
\boxed{
\text{Translation Equivariance}
}
$$

Conv3d 相对于 Conv2d 最核心的变化只是：

$$
(H,W)
\rightarrow
(D,H,W)
$$

如果：

$$
D=T
$$

那么可以进一步理解为：

$$
\boxed{
\text{二维空间局部建模}
\rightarrow
\text{时空联合局部建模}
}
$$

这也是后续理解 **3D VAE、Causal Conv3d 和视频生成模型** 的基础。

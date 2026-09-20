---
type: project
status: draft
created: 2026-09-20
updated: 2026-09-20
ai_generated: true
reviewed: false
---

# From geometric intuition to PyTorch

> [!warning]
> AI-authored exercises; pending human review. Completing the videos is the learner's reported background, not evidence that these exercises have been mastered.

These exercises connect concepts from *Essence of Linear Algebra* to computation. They are original practice problems, not official 3Blue1Brown exercises. Work in three rounds: **01–04, 05–08, 09–12**. Start with 01–03 and discuss your reasoning before continuing.

## Working method

For every exercise, submit three things:

1. **Predict:** a sketch, expected shape, or numerical prediction before running code.
2. **Compute:** a short PyTorch implementation, plus a residual or consistency check.
3. **Explain:** why the result follows geometrically, and what would change under a specified perturbation.

Use CPU and `torch.float64` (8 bytes per real element). All coordinates here are dimensionless. Treat mathematical vectors as column vectors, even when stored in a one-dimensional tensor. No autograd is needed for this set. Numerical agreement in examples is a check, not a proof of a general identity.

中文提示：每道题都要回答“这个 tensor 代表什么对象”和“这次运算在几何上做了什么”。运行成功只完成了 Compute。

## Run the starter

[main.py](main.py) contains one TODO per exercise, with the data prepared. From the Vault root, in PowerShell:

```powershell
& "$env:USERPROFILE\.venvs\torch\Scripts\python.exe" `
  "10_Projects/LLM Math Foundations/exercises/01-linear-algebra/01.01-geometry-to-tensors/problem/main.py" 1
```

Replace `1` with an exercise number. An untouched exercise prints `TODO`; this is expected. Implement the corresponding branch and finish that branch with `return`, leaving the final `NotImplementedError` for unfinished branches. Use the shared environment already created on this Windows host. Other machines only need Python and PyTorch. The setup does not require a GPU.

Reference code and explanation are in [solution/readme.md](../solution/readme.md). Try first, then compare one exercise at a time.

## API map

The following are lookup tools, not a memorization list. For `A` with shape `(m, n)` and `v` with shape `(n,)`:

| API / expression | Meaning or return value | Exercises |
|---|---|---|
| `torch.tensor(data, dtype=torch.float64)` | Build a floating-point tensor | All |
| `v.shape`, `A[:, j]`, `A[i, :]` | Shape, column, row | 01–04 |
| `torch.stack([u, v], dim=1)` | Store two 1-D vectors as columns | 01, 05, 09 |
| `A @ v`, `A @ B` | Matrix-vector or matrix-matrix multiplication | All |
| `A.T` | Transpose a 2-D matrix | 04, 06, 10 |
| `v[:, None]`, `v[None, :]` | Explicit column `(n,1)` / row `(1,n)` | 01, 04 |
| `torch.eye(n)`, `torch.zeros(n)` | Identity matrix and zero vector | 02, 11 |
| `torch.dot(u, v)` | Dot product of two 1-D vectors | 06 |
| `torch.outer(u, v)` | Matrix with entries $u_i v_j$ | 06 |
| `torch.linalg.vector_norm(v)` | Euclidean length by default | 06, 10 |
| `torch.linalg.det(A)` | Determinant of a square matrix | 05, 08 |
| `torch.linalg.matrix_rank(A)` | Numerical rank, affected by tolerance | 07 |
| `torch.linalg.solve(A, b)` | Solve a square nonsingular system | 08, 09 |
| `torch.linalg.lstsq(A, b).solution` | Least-squares solution | 10 |
| `torch.linalg.eig(A)` | Eigenvalues and column eigenvectors; complex output | 11 |
| `torch.linalg.eigh(S)` | Real eigenvalues/eigenvectors for real symmetric $S$ | Optional extension |
| `torch.linalg.cross(u, v)` | Cross product of 3-D vectors | 12 |
| `torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)` | Fail clearly when a numerical check disagrees | All |

API notes: `*` is elementwise multiplication; `@` is matrix multiplication. A tensor of shape `(n,)` has no row/column axis to transpose. Build matrices from floating-point data; linear algebra routines generally do not accept integer matrices. Avoid exact floating-point equality. When comparing with zero, an absolute tolerance matters.

Sources, accessed 2026-09-20: [PyTorch linear algebra](https://docs.pytorch.org/docs/2.14/linalg.html), [matmul shape rules](https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html), [stack](https://docs.pytorch.org/docs/2.14/generated/torch.stack.html). The `/stable/` pages resolved to 2.14 during preparation; these links pin the documentation used.

## 01 — Columns are transformed basis vectors

Given $\alpha=(2,1)^\top$, $\beta=(-1,2)^\top$, and coordinates $c=(3,-2)^\top$:

- Predict the quadrant of $v=3\alpha-2\beta$ and sketch the construction.
- Build $A=[\alpha\ \beta]$ using `stack`, then compute $v$ both as a linear combination and as `A @ c`.
- Verify `A @ e1 == alpha` and `A @ e2 == beta` numerically, where $e_1,e_2$ are the standard basis vectors.
- Compare `stack(..., dim=0)` with `dim=1`. Which matrix actually implements the requested transformation?

**Explain:** why does the first column describe the image of $e_1$? Do the coefficients refer to standard coordinates or to the basis $(\alpha,\beta)$?

## 02 — Construct a transformation from its action

Construct a counterclockwise quarter-turn $R$ and a horizontal shear $H$ with shear factor $2$, using only the images of $e_1,e_2$. Apply each to $v=(1,2)^\top$.

- Predict the output before assembling the matrices.
- Verify rotation preserves this vector's length. Does the shear preserve its length?
- Check linearity on $u=(2,-1)^\top$, $v=(1,2)^\top$ with coefficients $3,-2$.
- Try the translation $f(v)=v+(1,1)^\top$. Test $f(0)$ and explain why it is not a linear map on $\mathbb R^2$.

**APIs:** `stack`, `eye`, `@`, `vector_norm`.

## 03 — Composition and order

Using the same $R,H$ and $v=(1,2)^\top$, predict and compute:

$$
R(Hv),\qquad H(Rv),\qquad (RH)v.
$$

Explain which operation happens first in `R @ H @ v`. Give one vector for which the order makes a difference. Verify associativity for the first expression; do not confuse associativity with commutativity.

## 04 — One map, many vectors, and a rectangular matrix

Let $A=\begin{bmatrix}2&-1\\1&2\end{bmatrix}$. Store $e_1,e_2,(1,1)^\top$ as the rows of $X\in\mathbb R^{3\times2}$.

- Predict `A @ X.T` and its shape.
- Find an expression that stores the transformed vectors as rows. Check that its row $i$ agrees with `A @ X[i]`.
- Why is `X @ A` usually the wrong expression for this convention?
- Let $B=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}\in\mathbb R^{3\times2}$. Compute the image of $(2,-1)^\top$. What plane contains every possible output? Why is the ordinary square-matrix determinant unavailable?

中文提示：数据按 rows 存储并不改变数学上采用 column-vector convention；转置用于连接这两种表示。

## 05 — Oriented area and column replacement

Let $\alpha=(2,1)^\top$, $\beta=(1,2)^\top$, $A=[\alpha\ \beta]$.

- Compute the signed area directly as $\alpha_1\beta_2-\alpha_2\beta_1$, then check with `det`.
- Predict determinants after swapping columns, negating the first column, and replacing it by $\alpha+4\beta$.
- For $T=\operatorname{diag}(-2,1)$, verify $\det(TA)=\det(T)\det(A)$; distinguish ordinary area scaling from orientation reversal.
- Explain why sliding one edge parallel to the other preserves area.

Use `clone()` before replacing a column so the original tensor remains available.

## 06 — Dot product, projection, and a linear functional

Let $u=(2,1)^\top$ and $v=(1,3)^\top$.

- Derive the coefficient $t$ such that $p=tu$ and residual $r=v-p$ satisfy $u^\top r=0$.
- Compute $p,r$; verify orthogonality and $\lVert v\rVert^2=\lVert p\rVert^2+\lVert r\rVert^2$.
- Construct a matrix $P$ such that $Pv=p$. Verify $P^2=P$ and $P^\top=P$.
- Interpret $w\mapsto u^\top w$ as a map $\mathbb R^2\to\mathbb R$. Give a nonzero vector it sends to zero.
- Replace $u$ by $-u$. Which of $t,p,P$ change?

**APIs:** `dot`, `outer`, `@`, `vector_norm`.

## 07 — Rank, null space, and reachability

Let $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$.

- Predict its image and rank from its columns; check numerical rank.
- Find a nonzero $n$ with $An=0$ by hand, then verify.
- For $b=(3,6)^\top$, find one solution $x_0$ and verify three distinct solutions $x_0+tn$.
- For $b'=(3,7)^\top$, explain geometrically why no exact solution exists. Do not rely solely on a solver error.
- Extension: perturb the lower-right entry by a small positive $\epsilon$. Distinguish exact algebraic rank from numerical rank with a tolerance.

## 08 — Cramer's rule and solving systems

Let $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ and $b=(5,4)^\top$.

- Solve by hand; implement both determinant ratios and `torch.linalg.solve`.
- Check the residual $Ax-b$.
- Use $b=(-1,1)^\top$ as a second case. Explain a negative coordinate through oriented area.
- What changes if the columns of $A$ become collinear?

Use Cramer's rule for geometric understanding; use `solve` for the numerical system. Avoid computing an explicit inverse merely to solve one system.

## 09 — Change of basis

Let $P=\begin{bmatrix}1&1\\0&1\end{bmatrix}$ contain the new basis vectors in standard coordinates. Let $v=(3,1)^\top$ and let $R$ be the counterclockwise quarter-turn from 02.

- Find new coordinates $c$ using $Pc=v$; reconstruct $v$.
- Derive the representation $M$ of the same rotation in the new basis. Implement it using `solve(P, R @ P)`.
- Verify $P(Mc)=R(Pc)$.
- Is $\lVert c\rVert_2=\lVert v\rVert_2$? If not, did the vector's physical length change?

中文提示：change of basis 改变坐标描述；active transformation 改变向量。非正交基下，坐标的普通 Euclidean norm 不能直接当作原空间的长度。

## 10 — Least squares as projection

Let $A=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$ and $b=(1,1,3)^\top$.

- Show that $b$ is outside the column space of $A$.
- Compute $\hat x$ with `lstsq`, fitted vector $p=A\hat x$, and residual $r=b-p$.
- Check $A^\top r\approx0$. Explain this without derivatives.
- Compare $\lVert A(\hat x+\delta)-b\rVert_2$ with the fitted residual for three nonzero perturbations $\delta$ of your choice. Then explain why the orthogonality property establishes the general minimum.

This is a connection exercise extending the geometric projection picture. The columns here are independent, so the minimizing coordinates are unique.

## 11 — Eigenvectors are invariant directions

Let $A=\begin{bmatrix}2&1\\0&1\end{bmatrix}$.

- Predict the eigenvalues from the characteristic polynomial. Find one eigenvector for each by hand.
- Use `eig` and verify $AV=V\operatorname{diag}(\lambda)$, where eigenvectors are the columns of $V$.
- Apply the matrix three times to each eigenvector. Compare with scaling by $\lambda^3$.
- Explain why eigenvector signs, lengths, and returned ordering are not fixed answers to compare elementwise.
- Repeat `eig` for a quarter-turn. Why is there no nonzero real invariant line? Interpret the complex output as a limitation of the real-plane picture.

## 12 — Cross products and functions as vectors

**A. Volume:** let $u=(1,0,1)^\top$, $v=(0,2,0)^\top$, $w=(0,0,3)^\top$.

- Compute $u\times v$; check its dot product with each input.
- Check $(u\times v)\cdot w=\det[u\ v\ w]$. What does swapping $u,v$ change?

**B. Abstract vector space:** represent $p(t)=c_0+c_1t+c_2t^2$ by $c=(c_0,c_1,c_2)^\top$. Build the $3\times3$ matrix $D$ representing differentiation in the basis $(1,t,t^2)$.

- Derive each column by differentiating the corresponding basis function.
- Apply $D$ to $c=(1,-2,3)^\top$ and interpret the result as a polynomial.
- Verify $D^3=0$. Identify its null space and interpret it as a set of functions.

Optional Python skill: use `sum` to evaluate a polynomial at a chosen scalar. The variable $t$ here is a polynomial argument, not a tensor axis or training step.

## Feedback and completion

Send your code, predictions, outputs, and explanation for 01–03 first. A useful submission includes one initial mistake and how you detected it. We will check shape conventions, geometric meaning, and numerical validation separately. A passed script does not automatically mark the roadmap complete.

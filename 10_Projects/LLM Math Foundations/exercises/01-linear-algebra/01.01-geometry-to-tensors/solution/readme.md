---
type: project
status: draft
created: 2026-09-20
updated: 2026-09-20
ai_generated: true
reviewed: false
---

# Reference reasoning — Geometry to tensors

> [!warning]
> Spoilers. Try the [problems](../problem/readme.md) before reading this file. AI-authored reference reasoning remains pending human review. Numerical checks do not establish learner mastery or replace general proofs.

Run [main.py](main.py) with one number or `all`, using the same Python interpreter as the starter. The reference script checks its own calculations; it does not grade the student file.

Validation on 2026-09-20: all 12 reference computations and their assertions passed on Windows, Python 3.12.14, PyTorch 2.14.0+cpu, using float64. The untouched starter's 12 branches were also exercised and reported TODO as intended. This validates these examples, not a universal numerical-stability claim.

| Exercise | Reference result | Geometric reason |
|---|---|---|
| 01 | $v=(8,-1)^\top$ | Matrix columns are images of standard basis vectors; multiplication forms their weighted sum |
| 02 | $Rv=(-2,1)^\top$, $Hv=(5,2)^\top$ | A quarter-turn preserves length; a shear changes horizontal displacement according to vertical position |
| 03 | $RHv=(-2,5)^\top$, $HRv=(0,1)^\top$ | The second transformation acts on the output of the first; changing order changes that intermediate vector |
| 04 | Transformed rows: $(2,1),(-1,2),(1,3)$; $B(2,-1)^\top=(2,-1,1)^\top$ | Row storage requires `X @ A.T`; the rectangular map's image lies in $z=x+y$ |
| 05 | Determinants: $3,-3,-3,3,-6$ for original, swap, negate, slide, transform | Orientation and height determine signed area; sliding along the fixed side preserves height |
| 06 | $t=1$, $p=(2,1)^\top$, $r=(-1,2)^\top$ | The residual is perpendicular to the projection line |
| 07 | Rank $1$, null direction $(-2,1)^\top$, $x_0=(3,0)^\top$ | Adding a null vector does not change the image; reachable outputs satisfy $b_2=2b_1$ |
| 08 | Solutions $(2,1)^\top$ and $(-1,1)^\top$ | Each coordinate is a ratio of oriented areas, so negative coordinates are natural |
| 09 | $c=(2,1)^\top$, $M=\begin{bmatrix}-1&-2\\1&1\end{bmatrix}$ | $P$ converts new coordinates to standard coordinates; solve converts back |
| 10 | $\hat x=(4/3,4/3)^\top$, $r=(-1/3,-1/3,1/3)^\top$ | The fitted vector is the orthogonal projection onto the column plane |
| 11 | Eigenvalues $2,1$; directions $(1,0)^\top$, $(-1,1)^\top$ | These lines are preserved; a quarter-turn has eigenvalues $\pm i$ and no real invariant line |
| 12 | $u\times v=(-2,0,2)^\top$, oriented volume $6$; derivative coefficients $(-2,6,0)^\top$ | Cross products encode a perpendicular area vector; differentiation is linear on polynomial coefficients |

For 06, the condition $u^\top(v-tu)=0$ gives $t=(u^\top v)/(u^\top u)$ for nonzero $u$. Thus $P=uu^\top/(u^\top u)$. Negating $u$ negates $t$ but leaves $p$ and $P$ unchanged. The null space of the functional $w\mapsto u^\top w$ is the perpendicular line.

For 09, $M=P^{-1}RP$, implemented with `solve` rather than explicitly forming the inverse. Since the new basis is not orthonormal, physical squared length is $c^\top P^\top Pc$, not generally $c^\top c$.

For 10, let $r=b-A\hat x$ with $A^\top r=0$. Any alternative coordinates $\hat x+\delta$ give:

$$
\lVert A(\hat x+\delta)-b\rVert^2
=\lVert A\delta-r\rVert^2
=\lVert A\delta\rVert^2+\lVert r\rVert^2.
$$

The cross term vanishes because $r$ is orthogonal to the column space. This establishes the minimum for every $\delta$; a few numerical perturbations only illustrate it.

For 12, in the ordered basis $(1,t,t^2)$:

$$
D=\begin{bmatrix}0&1&0\\0&0&2\\0&0&0\end{bmatrix}.
$$

Every quadratic polynomial becomes zero after three derivatives. The null space consists of constant polynomials, represented by $(c_0,0,0)^\top$.

Numerical caveats: eigenvectors may change sign or ordering across implementations; verify the defining equations. Numerical rank depends on tolerance. The examples use small, well-scaled matrices and float64; this is not a general numerical-stability benchmark.

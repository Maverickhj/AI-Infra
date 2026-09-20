---
type: knowledge
status: seed
created: 2026-09-13
updated: 2026-09-13
domains:
  - mathematics
  - linear-algebra
source:
  - https://www.3blue1brown.com/lessons/cramers-rule/
ai_generated: true
reviewed: true
---

# Geometric intuition for Cramer's rule

> [!warning]
> This is an AI-organized learning note based on the discussion around 3Blue1Brown's Cramer's rule lesson. It is a conceptual record, not an independent proof review or a learning record showing mastery.

![[Geometric intuition for Cramers rule.assets/cramers-rule-area-ratio.svg]]

The diagram uses $\alpha=(3,1)$ and $\beta=(1,2)$. Since $b=2\alpha+\beta$, replacing $\alpha$ with $b$ doubles the oriented area while keeping $\beta$ fixed; this is the geometric reason that $x=2$.

## The determinant as oriented area

Let

$$
A=[\alpha\ \beta],
\qquad
\alpha,\beta\in\mathbb R^2.
$$

The two columns define the parallelogram

$$
P_A=\{s\alpha+t\beta\mid 0\le s,t\le1\}.
$$

Its ordinary area is

$$
\operatorname{area}(P_A)=|\det(A)|.
$$

The determinant itself is an **oriented area** (also called signed area):

$$
\det(A)=\det(\alpha,\beta).
$$

The sign depends on the ordered pair $(\alpha,\beta)$. If the rotation from $\alpha$ to $\beta$ follows the chosen positive orientation, the determinant is positive; reversing the columns changes the sign. If $\alpha$ and $\beta$ are collinear, the parallelogram is degenerate and the determinant is zero.

中文说明：严格说不是 parallelogram 这个形状本身“带符号”，而是带有边顺序的 ordered pair $(\alpha,\beta)$ 带有 orientation。普通面积永远非负；determinant 额外记录 orientation。

## Linear transformations and area scaling

For a linear transformation $T:\mathbb R^2\to\mathbb R^2$,

$$
T(P_A)=P_{TA}.
$$

Therefore,

$$
\det(TA)
=\det(T\alpha,T\beta)
=\det(T)\det(A).
$$

The scalar $\det(T)$ is the signed area scale factor of $T$. The ordinary area scale factor is $|\det(T)|$. A negative determinant means that $T$ reverses orientation; a zero determinant collapses the parallelogram into a lower-dimensional set.

## Cramer's rule as an area ratio

Suppose

$$
A
\begin{bmatrix}x\\y\end{bmatrix}
=b,
\qquad
b=x\alpha+y\beta.
$$

Assume $\det(A)\ne0$, so $\alpha$ and $\beta$ are linearly independent and the coordinates $(x,y)$ are unique.

To recover $x$, replace the first column $\alpha$ by $b$:

$$
\begin{aligned}
\det(b,\beta)
&=\det(x\alpha+y\beta,\beta)\\
&=x\det(\alpha,\beta)+y\det(\beta,\beta)\\
&=x\det(\alpha,\beta).
\end{aligned}
$$

Thus,

$$
\boxed{
x=\frac{\det(b,\beta)}{\det(\alpha,\beta)}
}.
$$

Geometrically, $y\beta$ only slides the vector along the direction of the fixed side $\beta$; it does not change the signed height above that side. The remaining component $x\alpha$ contributes exactly $x$ times the original oriented area.

Similarly,

$$
\begin{aligned}
\det(\alpha,b)
&=\det(\alpha,x\alpha+y\beta)\\
&=y\det(\alpha,\beta),
\end{aligned}
$$

so

$$
\boxed{
y=\frac{\det(\alpha,b)}{\det(\alpha,\beta)}
}.
$$

This is the two-dimensional form of Cramer's rule:

$$
x=\frac{\det[b\ \beta]}{\det[\alpha\ \beta]},
\qquad
y=\frac{\det[\alpha\ b]}{\det[\alpha\ \beta]}.
$$

The numerator is the oriented area after replacing one basis vector by $b$; the denominator is the oriented area of the original basis parallelogram.

## A small example

Let

$$
\alpha=\begin{bmatrix}2\\1\end{bmatrix},
\qquad
\beta=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
b=\begin{bmatrix}5\\4\end{bmatrix}.
$$

The original oriented area is

$$
\det(\alpha,\beta)=
\begin{vmatrix}2&1\\1&2\end{vmatrix}=3.
$$

Then

$$
\det(b,\beta)=
\begin{vmatrix}5&1\\4&2\end{vmatrix}=6,
\qquad
\det(\alpha,b)=
\begin{vmatrix}2&5\\1&4\end{vmatrix}=3.
$$

Therefore,

$$
x=\frac{6}{3}=2,
\qquad
y=\frac{3}{3}=1,
$$

which agrees with

$$
b=2\alpha+\beta.
$$

## Boundary of the picture

Cramer's rule requires $\det(A)\ne0$. Geometrically, the original parallelogram must have nonzero area. If the columns are collinear, they do not form a basis: the coordinate representation may be non-unique or may not exist, so the determinant ratios are undefined.

The same idea extends to $n$ dimensions: determinants measure oriented volume, and replacing one column by $b$ reveals the corresponding coordinate of $b$ in the basis formed by the columns of $A$.

Related roadmap entry: [[10_Projects/LLM Math Foundations/ROADMAP|LLM Math Foundations Roadmap]].

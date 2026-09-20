"""Reference computations with numerical checks; not a student autograder."""

import argparse
import torch


def t(data):
    return torch.tensor(data, dtype=torch.float64)


def close(actual, expected):
    if not isinstance(expected, torch.Tensor):
        expected = t(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def exercise(n):
    R, H = t([[0, -1], [1, 0]]), t([[1, 2], [0, 1]])
    if n == 1:
        alpha, beta, c = t([2, 1]), t([-1, 2]), t([3, -2])
        A = torch.stack([alpha, beta], dim=1)
        v = A @ c
        close(v, 3 * alpha - 2 * beta)
        close(v, [8, -1])
        close(A @ t([1, 0]), alpha)
        close(A @ t([0, 1]), beta)
        assert not torch.allclose(torch.stack([alpha, beta]) @ c, v)
        print("A, v:", A, v)
    elif n == 2:
        u, v = t([2, -1]), t([1, 2])
        close(R @ v, [-2, 1])
        close(H @ v, [5, 2])
        close(torch.linalg.vector_norm(R @ v), torch.linalg.vector_norm(v))
        assert not torch.allclose(torch.linalg.vector_norm(H @ v), torch.linalg.vector_norm(v))
        for A in [R, H]:
            close(A @ (3*u - 2*v), 3*(A @ u) - 2*(A @ v))
        offset = t([1, 1])
        assert not torch.allclose(t([0, 0]) + offset, t([0, 0]))
        print("Rotation and shear:", R @ v, H @ v)
    elif n == 3:
        v = t([1, 2])
        close(R @ (H @ v), (R @ H) @ v)
        close(R @ H @ v, [-2, 5])
        close(H @ R @ v, [0, 1])
        print("RHv, HRv:", R @ H @ v, H @ R @ v)
    elif n == 4:
        A, X = t([[2, -1], [1, 2]]), t([[1, 0], [0, 1], [1, 1]])
        Y = X @ A.T
        close(Y, (A @ X.T).T)
        close(Y, [[2, 1], [-1, 2], [1, 3]])
        for i in range(3):
            close(Y[i], A @ X[i])
        B = t([[1, 0], [0, 1], [1, 1]])
        output = B @ t([2, -1])
        close(output, [2, -1, 1])
        close(t([-1, -1, 1]) @ B, [0, 0])
        print("Row-stored outputs and rectangular output:", Y, output)
    elif n == 5:
        A, T = t([[2, 1], [1, 2]]), t([[-2, 0], [0, 1]])
        negated, slid = A.clone(), A.clone()
        negated[:, 0] *= -1
        slid[:, 0] += 4 * A[:, 1]
        determinants = torch.stack([torch.linalg.det(M) for M in [A, A[:, [1, 0]], negated, slid, T @ A]])
        close(determinants, [3, -3, -3, 3, -6])
        close(torch.linalg.det(T @ A), torch.linalg.det(T)*torch.linalg.det(A))
        close(A[0, 0]*A[1, 1]-A[1, 0]*A[0, 1], determinants[0])
        print("Signed areas:", determinants)
    elif n == 6:
        u, v = t([2, 1]), t([1, 3])
        coefficient = torch.dot(u, v) / torch.dot(u, u)
        p, P = coefficient*u, torch.outer(u, u)/torch.dot(u, u)
        r = v-p
        close(p, [2, 1])
        close(r, [-1, 2])
        close(torch.dot(u, r), 0.)
        close(torch.dot(v, v), torch.dot(p, p)+torch.dot(r, r))
        close(P @ v, p)
        close(P @ P, P)
        close(P.T, P)
        close(torch.outer(-u, -u)/torch.dot(-u, -u), P)
        close(torch.dot(-u, v)/torch.dot(-u, -u), -coefficient)
        print("Projection, residual, matrix:", p, r, P)
    elif n == 7:
        A, null, x0, b = t([[1, 2], [2, 4]]), t([-2, 1]), t([3, 0]), t([3, 6])
        assert torch.linalg.matrix_rank(A).item() == 1
        close(A @ null, [0, 0])
        for scalar in [-2., 0., 3.]:
            close(A @ (x0+scalar*null), b)
        normal, outside = t([-2, 1]), t([3, 7])
        close(normal @ A, [0, 0])
        close(normal @ outside, 1.)
        print("Null direction:", null, "outside-image certificate:", normal @ outside)
    elif n == 8:
        A = t([[2, 1], [1, 2]])
        for b, expected in [(t([5, 4]), t([2, 1])), (t([-1, 1]), t([-1, 1]))]:
            first, second = A.clone(), A.clone()
            first[:, 0], second[:, 1] = b, b
            cramer = torch.stack([torch.linalg.det(first), torch.linalg.det(second)]) / torch.linalg.det(A)
            solved = torch.linalg.solve(A, b)
            close(cramer, solved)
            close(solved, expected)
            close(A @ solved-b, [0, 0])
            print("Target, coordinates:", b, solved)
    elif n == 9:
        P, v = t([[1, 1], [0, 1]]), t([3, 1])
        c = torch.linalg.solve(P, v)
        M = torch.linalg.solve(P, R @ P)
        close(c, [2, 1])
        close(M, [[-1, -2], [1, 1]])
        close(P @ c, v)
        close(P @ (M @ c), R @ (P @ c))
        close(c @ P.T @ P @ c, torch.dot(v, v))
        assert not torch.allclose(torch.linalg.vector_norm(c), torch.linalg.vector_norm(v))
        print("New coordinates and matrix:", c, M)
    elif n == 10:
        A, b = t([[1, 0], [0, 1], [1, 1]]), t([1, 1, 3])
        x = torch.linalg.lstsq(A, b).solution
        residual = b-A @ x
        close(x, [4/3, 4/3])
        close(residual, [-1/3, -1/3, 1/3])
        close(A.T @ residual, [0, 0])
        for delta in [t([1, 0]), t([0, -1]), t([.1, -.2])]:
            perturbed = A @ (x+delta)-b
            close(torch.dot(perturbed, perturbed), torch.dot(residual, residual)+torch.dot(A @ delta, A @ delta))
            assert torch.linalg.vector_norm(perturbed) > torch.linalg.vector_norm(residual)
        print("Coordinates, projection, residual:", x, A @ x, residual)
    elif n == 11:
        A = t([[2, 1], [0, 1]])
        eigenvalues, V = torch.linalg.eig(A)
        Ac = A.to(V.dtype)
        close(Ac @ V, V @ torch.diag(eigenvalues))
        close(torch.linalg.matrix_power(Ac, 3) @ V, V @ torch.diag(eigenvalues**3))
        close(A @ t([1, 0]), 2*t([1, 0]))
        close(A @ t([-1, 1]), t([-1, 1]))
        rotation_values, rotation_vectors = torch.linalg.eig(R)
        close(R.to(rotation_vectors.dtype) @ rotation_vectors, rotation_vectors @ torch.diag(rotation_values))
        close(rotation_values.real, [0, 0])
        close(rotation_values.imag.abs(), [1, 1])
        print("Eigenvalues and quarter-turn eigenvalues:", eigenvalues, rotation_values)
    elif n == 12:
        u, v, w = t([1, 0, 1]), t([0, 2, 0]), t([0, 0, 3])
        cross = torch.linalg.cross(u, v)
        close(cross, [-2, 0, 2])
        close(torch.dot(cross, u), 0.)
        close(torch.dot(cross, v), 0.)
        close(torch.dot(cross, w), torch.linalg.det(torch.stack([u, v, w], dim=1)))
        close(torch.dot(cross, w), 6.)
        close(torch.linalg.cross(v, u), -cross)
        D, c = t([[0, 1, 0], [0, 0, 2], [0, 0, 0]]), t([1, -2, 3])
        close(D @ c, [-2, 6, 0])
        close(torch.linalg.matrix_power(D, 3), torch.zeros_like(D))
        close(D @ t([1, 0, 0]), [0, 0, 0])
        print("Area vector, derivative coefficients:", cross, D @ c)
    else:
        raise ValueError("Exercise number must be between 1 and 12")
    print(f"PASS {n:02d}: reference numerical checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=[str(i) for i in range(1, 13)] + ["all"])
    args = parser.parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)
    selected = range(1, 13) if args.exercise == "all" else [int(args.exercise)]
    for number in selected:
        exercise(number)

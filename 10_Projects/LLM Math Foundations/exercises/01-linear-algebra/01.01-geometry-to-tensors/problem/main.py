"""Student workspace. Usage: python main.py 1 (or 'all'). AI-authored draft."""

import argparse
import torch
import numpy


def tensor(data):
    return torch.tensor(data, dtype=torch.float64)


def exercise(number):
    # Add your prediction in a comment BEFORE computing each exercise.
    # Keep the branches independent so any exercise can run on its own.
    # Finish each completed branch with `return`; leave the final raise in place.
    if number == 1:
        alpha, beta, c = tensor([2, 1]), tensor([-1, 2]), tensor([3, -2])
        # TODO: form A, compute v in two ways, verify the basis images.
        v1 = 3 * alpha + 2 * beta
        A = torch.stack([alpha, beta], dim=1)
        v2 = A @ c
        print("v1:", v1)
        print("v2:", v2)
        e1 = torch.tensor([1, 0], dtype=torch.float64)
        e2 = torch.tensor([0, 1], dtype=torch.float64)
        print("A @ e1:", A @ e1)
        print("A @ e2:", A @ e2)
        
        return
    elif number == 2:
        e1, e2 = tensor([1, 0]), tensor([0, 1])
        v, u = tensor([1, 2]), tensor([2, -1])
        # TODO: construct R/H from basis images; test length and linearity.
    elif number == 3:
        R, H = tensor([[0, -1], [1, 0]]), tensor([[1, 2], [0, 1]])
        v = tensor([1, 2])
        # TODO: compare both orders and verify associativity.
    elif number == 4:
        A = tensor([[2, -1], [1, 2]])
        X = tensor([[1, 0], [0, 1], [1, 1]])
        B = tensor([[1, 0], [0, 1], [1, 1]])
        # TODO: transform row-stored vectors; investigate the image of B.
    elif number == 5:
        alpha, beta = tensor([2, 1]), tensor([1, 2])
        T = tensor([[-2, 0], [0, 1]])
        # TODO: compare signed areas under column operations and T.
    elif number == 6:
        u, v = tensor([2, 1]), tensor([1, 3])
        # TODO: derive t; compute p/r/P; check projection properties.
    elif number == 7:
        A = tensor([[1, 2], [2, 4]])
        b, b_outside = tensor([3, 6]), tensor([3, 7])
        # TODO: find rank, null direction and a family of solutions.
    elif number == 8:
        A = tensor([[2, 1], [1, 2]])
        targets = [tensor([5, 4]), tensor([-1, 1])]
        # TODO: Cramer ratios and solve, with residual checks for both targets.
    elif number == 9:
        P, R = tensor([[1, 1], [0, 1]]), tensor([[0, -1], [1, 0]])
        v = tensor([3, 1])
        # TODO: coordinates c and representation M; reconstruct both paths.
    elif number == 10:
        A, b = tensor([[1, 0], [0, 1], [1, 1]]), tensor([1, 1, 3])
        # TODO: least squares, orthogonal residual, perturbation comparisons.
    elif number == 11:
        A, R = tensor([[2, 1], [0, 1]]), tensor([[0, -1], [1, 0]])
        # TODO: eigenpairs, repeated action, and the quarter-turn's eigenvalues.
    elif number == 12:
        u, v, w = tensor([1, 0, 1]), tensor([0, 2, 0]), tensor([0, 0, 3])
        c = tensor([1, -2, 3])
        # TODO: cross product/volume; build differentiation operator D.
    else:
        raise ValueError("Exercise number must be between 1 and 12")
    raise NotImplementedError("Complete the selected branch and end it with return")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=[str(i) for i in range(1, 13)] + ["all"])
    args = parser.parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)
    selected = range(1, 13) if args.exercise == "all" else [int(args.exercise)]
    for number in selected:
        print(f"Exercise {number:02d}")
        try:
            exercise(number)
        except NotImplementedError as error:
            print(f"TODO: {error}")

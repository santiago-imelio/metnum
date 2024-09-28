defmodule Metnum.Matrix do
  @moduledoc """
  Collection of functions to create interesting matrices.
  """

  @doc """
  The Dorr matrix is A diagonally dominant, tridiagonal, M-matrix.
  It is ill conditioned for small values of the (positive) parameter, theta.
  The columns of the inverse of this matrix vary greatly in norm.

  The Gauss-Seidel and Jacobi iterative methods for solving `Ax = b` both converge.

  Source: https://people.math.sc.edu/burkardt/py_src/test_mat/dorr.py
  """
  def dorr(n, theta) do
    np1_r8 = n + 1

    for i <- 0..(n - 1) do
      for j <- 0..(n - 1) do
        if i + 1 <= div(n + 1, 2) do
          cond do
            j == i - 1 -> -theta * np1_r8 ** 2
            j == i -> 2.0 * theta * np1_r8 ** 2 + 0.5 * np1_r8 - (i + 1)
            j == i + 1 -> -theta * np1_r8 ** 2 - 0.5 * np1_r8 + (i + 1)
            true -> 0
          end
        else
          cond do
            j == i - 1 -> -theta * np1_r8 ** 2 + 0.5 * np1_r8 - (i + 1)
            j == i -> 2.0 * theta * np1_r8 ** 2 - 0.5 * np1_r8 + (i + 1)
            j == i + 1 -> -theta * np1_r8 ** 2
            true -> 0
          end
        end
      end
    end
    |> Nx.tensor()
  end
end

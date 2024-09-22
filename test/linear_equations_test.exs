defmodule Metnum.LinearEquationsTest do
  use Metnum.Case

  alias Metnum.LinearEquations, as: LE

  describe "jacobi solver" do
    test "2 x 2 system of equations #1" do
      a = Nx.tensor([[2, -1], [1, 3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      opts = [
        max_epochs: 8,
        tolerance: 0.01,
        solver: :jacobi
      ]

      x_0 = Nx.tensor([0.0, 0.0])

      seq = LE.solve(a, b, x_0, opts)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[1], Nx.tensor([2.5, 2.333]), opts[:tolerance])
      assert equal_within_epsilon(seq[2], Nx.tensor([3.6667, 1.5]), opts[:tolerance])
      assert equal_within_epsilon(seq[3], Nx.tensor([3.25, 1.1111]), opts[:tolerance])
      assert equal_within_epsilon(seq[4], Nx.tensor([3.0556, 1.25]), opts[:tolerance])
      assert equal_within_epsilon(seq[5], Nx.tensor([3.125, 1.3148]), opts[:tolerance])
      assert equal_within_epsilon(seq[6], Nx.tensor([3.1574, 1.2917]), opts[:tolerance])
      assert equal_within_epsilon(seq[7], solution, opts[:tolerance])
    end

    test "2 x 2 system of equations #2" do
      a = Nx.tensor([[2, 1], [5, 7]])
      b = Nx.tensor([11, 13])

      solution = Nx.tensor([7.111, -3.222])

      opts = [
        max_epochs: 30,
        solver: :jacobi
      ]

      x_0 = Nx.tensor([1.0, 1.0])

      seq = LE.solve(a, b, x_0, opts)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[opts[:max_epochs] - 1], solution, 0.001)
    end
  end

  describe "gauss_seidel solver" do
    test "2 x 2 system of equations #1" do
      a = Nx.tensor([[2, -1], [1, 3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      opts = [
        max_epochs: 5,
        tolerance: 0.01,
        solver: :gauss_seidel
      ]

      x_0 = Nx.tensor([0.0, 0.0])

      seq = LE.solve(a, b, x_0, opts)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[1], Nx.tensor([2.5, 1.5]), opts[:tolerance])
      assert equal_within_epsilon(seq[2], Nx.tensor([3.25, 1.25]), opts[:tolerance])
      assert equal_within_epsilon(seq[3], Nx.tensor([3.125, 1.292]), opts[:tolerance])
      assert equal_within_epsilon(seq[4], solution, opts[:tolerance])
    end

    test "2 x 2 system of equations #2" do
      a = Nx.tensor([[2, 1], [5, 7]])
      b = Nx.tensor([11, 13])

      solution = Nx.tensor([7.111, -3.222])

      opts = [
        max_epochs: 9,
        solver: :gauss_seidel
      ]

      x_0 = Nx.tensor([1.0, 1.0])

      seq = LE.solve(a, b, x_0, opts)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[opts[:max_epochs] - 1], solution, 0.001)
    end
  end

  describe "sor solver" do
    test "2 x 2 system of equations #1, w = 1 matches gauss-seidel" do
      a = Nx.tensor([[2, -1], [1, 3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      opts = [
        tolerance: 0.01,
        max_epochs: 5,
        solver: :sor,
        omega: 0.9999
      ]

      x_0 = Nx.tensor([0.0, 0.0])

      seq = LE.solve(a, b, x_0, opts)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[1], Nx.tensor([2.5, 1.5]), opts[:tolerance])
      assert equal_within_epsilon(seq[2], Nx.tensor([3.25, 1.25]), opts[:tolerance])
      assert equal_within_epsilon(seq[3], Nx.tensor([3.125, 1.292]), opts[:tolerance])
      assert equal_within_epsilon(seq[4], solution, opts[:tolerance])
    end

    test "2 x 2 system of equations #2" do
      a = Nx.tensor([[2, 1], [5, 7]])
      b = Nx.tensor([11, 13])

      solution = Nx.tensor([7.111, -3.222])

      opts = [
        max_epochs: 9,
        solver: :sor,
        omega: 0.9999
      ]

      x_0 = Nx.tensor([1.0, 1.0])

      seq = LE.solve(a, b, x_0, opts)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[opts[:max_epochs] - 1], solution, 0.001)
    end

    test "4 x 4 system of equations" do
      a =
        Nx.tensor([
          [4, -1, -6, 0],
          [-5, -4, 10, 8],
          [0, 9, 4, -2],
          [1, 0, -7, 5]
        ])

      b = Nx.tensor([2, 21, -12, -6])

      solution = Nx.tensor([3.0, -2.0, 2.0, 1.0])
      x_0 = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      opts = [
        max_epochs: 30,
        tolerance: 0.001,
        solver: :sor,
        omega: 0.5
      ]

      seq = LE.solve(a, b, x_0, opts)

      assert equal_within_epsilon(
               seq[1],
               Nx.tensor([0.25, -2.781, 1.629, 0.515]),
               opts[:tolerance]
             )

      assert equal_within_epsilon(
               seq[2],
               Nx.tensor([1.249, -2.245, 1.969, 0.911]),
               opts[:tolerance]
             )

      assert equal_within_epsilon(
               seq[3],
               Nx.tensor([2.070, -1.670, 1.590, 0.762]),
               opts[:tolerance]
             )

      assert equal_within_epsilon(seq[29], solution, opts[:tolerance])
    end
  end
end

defmodule Metnum.SOETest do
  use Metnum.Case

  alias Metnum.SOE

  describe "#jacobi/4" do
    test "2 x 2 system of equations #1" do
      a = Nx.tensor([[2, -1], [1, 3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      tol = 0.01
      epochs = 8
      x_0 = Nx.tensor([0.0, 0.0])

      seq = SOE.jacobi(a, b, x_0, max_epochs: epochs, tolerance: tol)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[1], Nx.tensor([2.5, 2.333]), tol)
      assert equal_within_epsilon(seq[2], Nx.tensor([3.6667, 1.5]), tol)
      assert equal_within_epsilon(seq[3], Nx.tensor([3.25, 1.1111]), tol)
      assert equal_within_epsilon(seq[4], Nx.tensor([3.0556, 1.25]), tol)
      assert equal_within_epsilon(seq[5], Nx.tensor([3.125, 1.3148]), tol)
      assert equal_within_epsilon(seq[6], Nx.tensor([3.1574, 1.2917]), tol)
      assert equal_within_epsilon(seq[7], solution, tol)
    end

    test "2 x 2 system of equations #2" do
      a = Nx.tensor([[2, 1], [5, 7]])
      b = Nx.tensor([11, 13])

      solution = Nx.tensor([7.111, -3.222])

      epochs = 30
      x_0 = Nx.tensor([1.0, 1.0])

      seq = SOE.jacobi(a, b, x_0, max_epochs: epochs)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[epochs - 1], solution, 0.001)
    end
  end

  describe "#gauss_seidel/4" do
    test "2 x 2 system of equations #1" do
      a = Nx.tensor([[2, -1], [1, 3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      tol = 0.01
      epochs = 5
      x_0 = Nx.tensor([0.0, 0.0])

      seq = SOE.gauss_seidel(a, b, x_0, max_epochs: epochs, tolerance: tol)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[1], Nx.tensor([2.5, 1.5]), tol)
      assert equal_within_epsilon(seq[2], Nx.tensor([3.25, 1.25]), tol)
      assert equal_within_epsilon(seq[3], Nx.tensor([3.125, 1.292]), tol)
      assert equal_within_epsilon(seq[4], solution, tol)
    end

    test "2 x 2 system of equations #2" do
      a = Nx.tensor([[2, 1], [5, 7]])
      b = Nx.tensor([11, 13])

      solution = Nx.tensor([7.111, -3.222])

      epochs = 9
      x_0 = Nx.tensor([1.0, 1.0])

      seq = SOE.gauss_seidel(a, b, x_0, max_epochs: epochs)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[epochs - 1], solution, 0.001)
    end
  end

  describe "#sor/4" do
    test "2 x 2 system of equations #1, w = 1 matches gauss-seidel" do
      a = Nx.tensor([[2, -1], [1, 3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      tol = 0.01
      epochs = 5
      x_0 = Nx.tensor([0.0, 0.0])
      w = 0.9999

      seq = SOE.sor(a, b, x_0, max_epochs: epochs, tolerance: tol, omega: w)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[1], Nx.tensor([2.5, 1.5]), tol)
      assert equal_within_epsilon(seq[2], Nx.tensor([3.25, 1.25]), tol)
      assert equal_within_epsilon(seq[3], Nx.tensor([3.125, 1.292]), tol)
      assert equal_within_epsilon(seq[4], solution, tol)
    end

    test "2 x 2 system of equations #2" do
      a = Nx.tensor([[2, 1], [5, 7]])
      b = Nx.tensor([11, 13])

      solution = Nx.tensor([7.111, -3.222])

      epochs = 9
      x_0 = Nx.tensor([1.0, 1.0])
      w = 0.9999

      seq = SOE.sor(a, b, x_0, max_epochs: epochs, omega: w)

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[epochs - 1], solution, 0.001)
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
      epochs = 30
      w = 0.5
      tol = 0.001

      seq = SOE.sor(a, b, x_0, max_epochs: epochs, tolerance: tol, omega: w)
      assert equal_within_epsilon(seq[1], Nx.tensor([0.25, -2.781, 1.629, 0.515]), tol)
      assert equal_within_epsilon(seq[2], Nx.tensor([1.249, -2.245, 1.969, 0.911]), tol)
      assert equal_within_epsilon(seq[3], Nx.tensor([2.070, -1.670, 1.590, 0.762]), tol)
      assert equal_within_epsilon(seq[29], solution, tol)
    end
  end
end

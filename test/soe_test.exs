defmodule Metnum.SOETest do
  use Metnum.Case

  alias Metnum.SOE

  describe "#jacobi/5" do
    test "2 x 2 system of equations #1" do
      a = Nx.tensor([[2, -1], [1,3]])
      b = Nx.tensor([5, 7])

      solution = Nx.tensor([22 / 7, 9 / 7])

      tol = 0.01
      epochs = 8
      x_0 = Nx.tensor([0.0, 0.0])

      seq = SOE.jacobi(a, b, x_0, [max_epochs: epochs, tolerance: tol])

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

      seq = SOE.jacobi(a, b, x_0, [max_epochs: epochs])

      assert seq[0] == x_0
      assert equal_within_epsilon(seq[epochs - 1], solution, 0.001)
    end
  end
end

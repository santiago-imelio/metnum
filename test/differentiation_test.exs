defmodule Metnum.DifferentiationTest do
  use Metnum.Case

  alias Metnum.Differentiation, as: D

  describe "#central_difference/3" do
    test "1 variable function" do
      eps = 0.0001

      # f(x) = x^2
      func1 = &(Nx.pow(&1, 2))

      # f(x) = x^3 + 3x^2 + 5
      func2 = fn x ->
        term1 = Nx.pow(x, 3)
        term2 = Nx.multiply(func1.(x), 3)
        term3 = 5

        term1
        |> Nx.add(term2)
        |> Nx.add(term3)
      end

      cd1 = D.central_difference(func1, 0, 0.1)
      expected1 = Nx.tensor(0.0)

      cd2 = D.central_difference(func2, 0, 0.1)
      expected2 = Nx.tensor(0.01)

      assert cd1 === expected1
      assert equal_within_epsilon(cd2, expected2, eps)
    end
  end
end

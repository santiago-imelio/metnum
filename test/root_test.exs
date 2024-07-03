defmodule Metnum.RootTest do
  use Metnum.Case

  alias Metnum.Root, as: R

  describe "#newton_raphson/3" do
    test "1 variable function" do
      eps = 0.00001

      # f(x) = x^2 - 3
      func1 = &Nx.add(Nx.pow(&1, 2), -3)

      # f(x) = x^2 + x - 2
      func2 = &Nx.add(Nx.pow(&1,2), Nx.add(&1, -2))

      epochs1 = 5
      func1_root_seq1 = R.newton_raphson(func1, 0.5, epochs1)
      func1_root_seq2 = R.newton_raphson(func1, -0.5, epochs1)

      assert equal_within_epsilon(func1_root_seq1[epochs1], Nx.sqrt(3), eps)
      assert equal_within_epsilon(func1_root_seq2[epochs1], Nx.negate(Nx.sqrt(3)), eps)

      epochs2 = 10
      func2_root_seq1 = R.newton_raphson(func2, 0, epochs2)
      func2_root_seq2 = R.newton_raphson(func2, -1, epochs2)

      assert equal_within_epsilon(func2_root_seq1[epochs2], Nx.tensor(1), eps)
      assert equal_within_epsilon(func2_root_seq2[epochs2], Nx.tensor(-2), eps)
    end
  end
end

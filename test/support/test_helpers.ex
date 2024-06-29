defmodule Metnum.TestHelpers do
  @doc """
  Returns true if the absolute value of the difference is
  less than the machine epsilon.
  """
  def equal_within_epsilon(a, b, type \\ :f32) do
    Nx.subtract(a, b)
    |> Nx.abs()
    |> Nx.less(Nx.Constants.epsilon(type))
  end
end

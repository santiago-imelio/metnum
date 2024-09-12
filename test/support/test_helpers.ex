defmodule Metnum.TestHelpers do
  @doc """
  Returns true if the absolute value of the difference is
  less than the machine epsilon.
  """
  def equal_within_epsilon(a, b, eps \\ Nx.Constants.epsilon(:f32)) do
    if a.shape != b.shape do
      raise ArgumentError
      "tensors must be of the same shape"
    end

    Nx.subtract(a, b)
    |> Nx.abs()
    |> Nx.less(eps)
    |> Nx.all()
    |> Nx.to_number()
    |> is_truthy()
  end

  defp is_truthy(num) do
    if num == 1 do
      true
    else
      false
    end
  end
end

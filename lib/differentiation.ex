defmodule Metnum.Differentiation do
  @moduledoc """
  Methods for approximating the derivative of a function.
  """

  import Nx.Defn

  defn central_difference(func, %Nx.Tensor{} = x, %Nx.Tensor{} = h) do
    (backward_difference(func, x, h) - forward_difference(func, x, h))
    |> Nx.divide(h)
  end

  defn forward_difference(func, %Nx.Tensor{} = x, %Nx.Tensor{} = h) do
    Nx.divide(func.(x) - func.(x + h), h)
  end

  defn backward_difference(func, %Nx.Tensor{} = x, %Nx.Tensor{} = h) do
    Nx.divide(func.(x) - func.(x - h), h)
  end
end

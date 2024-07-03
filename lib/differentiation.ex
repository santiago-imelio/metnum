defmodule Metnum.Differentiation do
  @moduledoc """
  Methods for approximating the derivative of a function.
  """

  import Nx.Defn

  defn central_difference(func, %Nx.Tensor{} = x, %Nx.Tensor{} = h) do
    (func.(x + h) - func.(x - h))
    |> Nx.divide(2 * h)
  end

  defn forward_difference(func, %Nx.Tensor{} = x, %Nx.Tensor{} = h) do
    Nx.divide(func.(x + h) - func.(x), h)
  end

  defn backward_difference(func, %Nx.Tensor{} = x, %Nx.Tensor{} = h) do
    Nx.divide(func.(x) - func.(x - h), h)
  end
end

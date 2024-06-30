defmodule Metnum.Root do
  @moduledoc """
  Methods for approximating the real-valued roots of a function.
  """

  import Nx.Defn

  def newton_raphson(func, x_0, epochs) do
    newton_raphson_aux(func, Nx.tensor([x_0]), 0, epochs)
  end

  defp newton_raphson_aux(_func, seq, curr_epoch, epochs) when curr_epoch == epochs do
    seq
  end

  defp newton_raphson_aux(func, seq, curr_epoch, epochs) do
    seq_curr = newton_raphson_step(func, seq, curr_epoch)
    newton_raphson_aux(func, seq_curr, curr_epoch + 1, epochs)
  end

  defnp newton_raphson_step(func, seq, curr_epoch) do
    x_prev = seq[curr_epoch]
    func_grad = grad(func)
    x_curr = x_prev - func.(x_prev) / func_grad.(x_prev)

    Nx.concatenate([seq, Nx.broadcast(x_curr, {1})])
  end
end

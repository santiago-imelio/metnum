defmodule Metnum.SOE do
  @moduledoc """
  Iterative methods for resolving a system of equations.
  """

  import Nx.Defn

  jacobi_opts_schema = [
    tolerance: [
      doc: "The given tolerance of the solution of the system.",
      default: Nx.Constants.epsilon(:f32)
    ],
    max_epochs: [
      doc: "Maximum of iterations for Jacobi method.",
      default: 100
    ]
  ]

  @jacobi_opts_schema NimbleOptions.new!(jacobi_opts_schema)

  def jacobi(
        %Nx.Tensor{} = a,
        %Nx.Tensor{} = b,
        %Nx.Tensor{} = x_0,
        opts \\ []
      ) do
    opts = NimbleOptions.validate!(opts, @jacobi_opts_schema)

    jacobi_n(a, b, x_0, opts)
  end

  defnp jacobi_n(a, b, x, opts) do
    {n} = Nx.shape(x)

    max_epochs = opts[:max_epochs]
    tol = opts[:tolerance]

    # sequence filler tensor
    sol_seq = Nx.broadcast(:nan, {max_epochs, n})

    # put x_0 in the sequence
    sol_seq = Nx.put_slice(sol_seq, [0, 0], Nx.broadcast(x, {1, n}))

    {sequence, _} =
      while {sol_seq, {i = 0, a, b}}, not end_jacobi?(sol_seq, i, tol, max_epochs) do
        x_new = jacobi_step(a, b, sol_seq[i])

        {Nx.put_slice(sol_seq, [i + 1, 0], x_new), {i + 1, a, b}}
      end

    sequence
  end

  defnp end_jacobi?(sequence, i, tol, max_epochs) do
    if i != 0 do
      prev_solution = sequence[i - 1]
      curr_solution = sequence[i]

      i > max_epochs or l2_norm(prev_solution, curr_solution) < tol
    else
      0
    end
  end

  deftransformp l2_norm(x, y) do
    Nx.subtract(x, y) |> Nx.LinAlg.norm(ord: 2)
  end

  defnp jacobi_step(a, b, x_prev) do
    {n} = Nx.shape(x_prev)

    x = Nx.broadcast(0.0, {n})

    {x_new, _} =
      while {x, {j = 0, a, b, x_prev}}, j < n do
        # compute terms of the numerator sum
        sum = Nx.broadcast(0.0, {1})

        {sum_til_j, _} =
          while {sum, {k = 0, j, a, x_prev}}, k < j do
            {Nx.add(a[j][k] * x_prev[k], sum), {k + 1, j, a, x_prev}}
          end

        {sum_after_j, _} =
          while {sum, {k = j + 1, j, a, x_prev}}, k < n do
            {Nx.add(a[j][k] * x_prev[k], sum), {k + 1, j, a, x_prev}}
          end

        x_j = (b[j] - sum_til_j - sum_after_j) / a[j][j]

        {Nx.put_slice(x, [j], Nx.broadcast(x_j, {1})), {j + 1, a, b, x_prev}}
      end

    Nx.broadcast(x_new, {1, n})
  end

  gauss_seidel_opts_schema = [
    tolerance: [
      doc: "The given tolerance of the solution of the system.",
      default: Nx.Constants.epsilon(:f32)
    ],
    max_epochs: [
      doc: "Maximum of iterations for Gauss-Seidel method.",
      default: 100
    ]
  ]

  @gauss_seidel_opts_schema NimbleOptions.new!(gauss_seidel_opts_schema)

  def gauss_seidel(
        %Nx.Tensor{} = a,
        %Nx.Tensor{} = b,
        %Nx.Tensor{} = x_0,
        opts \\ []
      ) do
    opts = NimbleOptions.validate!(opts, @gauss_seidel_opts_schema)

    gauss_seidel_n(a, b, x_0, opts)
  end

  defnp gauss_seidel_n(a, b, x, opts) do
    {n} = Nx.shape(x)

    max_epochs = opts[:max_epochs]
    tol = opts[:tolerance]

    # sequence filler tensor
    sol_seq = Nx.broadcast(:nan, {max_epochs, n})

    # put x_0 in the sequence
    sol_seq = Nx.put_slice(sol_seq, [0, 0], Nx.broadcast(x, {1, n}))

    {sequence, _} =
      while {sol_seq, {i = 0, a, b}}, not end_gauss_seidel?(sol_seq, i, tol, max_epochs) do
        x_new = gauss_seidel_step(a, b, sol_seq[i])

        {Nx.put_slice(sol_seq, [i + 1, 0], x_new), {i + 1, a, b}}
      end

    sequence
  end

  defnp end_gauss_seidel?(sequence, i, tol, max_epochs) do
    if i != 0 do
      prev_solution = sequence[i - 1]
      curr_solution = sequence[i]

      i > max_epochs or l2_norm(prev_solution, curr_solution) < tol
    else
      0
    end
  end

  defnp gauss_seidel_step(a, b, x_prev) do
    {n} = Nx.shape(x_prev)

    x = Nx.broadcast(0.0, {n})

    {x_new, _} =
      while {x, {j = 0, a, b, x_prev}}, j < n do
        # compute terms of the numerator sum
        sum = Nx.broadcast(0.0, {1})

        {sum_til_j, _} =
          while {sum, {k = 0, j, a, x}}, k < j do
            {Nx.add(a[j][k] * x[k], sum), {k + 1, j, a, x}}
          end

        {sum_after_j, _} =
          while {sum, {k = j + 1, j, a, x_prev}}, k < n do
            {Nx.add(a[j][k] * x_prev[k], sum), {k + 1, j, a, x_prev}}
          end

        x_j = (b[j] - sum_til_j - sum_after_j) / a[j][j]

        {Nx.put_slice(x, [j], Nx.broadcast(x_j, {1})), {j + 1, a, b, x_prev}}
      end

    Nx.broadcast(x_new, {1, n})
  end
end

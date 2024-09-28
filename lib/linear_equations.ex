defmodule Metnum.LinearEquations do
  alias Metnum.LinearEquations.Solvers

  solve_opts_schema = [
    tolerance: [
      doc: "The given tolerance of the solution of the system.",
      default: Nx.Constants.epsilon(:f32)
    ],
    max_epochs: [
      doc: "Maximum of iterations for the iterative method.",
      default: 100
    ],
    solver: [
      doc: "The iterative method to use.",
      type: {:in, [:jacobi, :gauss_seidel, :sor]},
      default: :jacobi
    ],
    omega: [
      doc: """
      The SOR relaxation factor. Only applicable when using SOR method.
      For convergence, 0 < w < 2.
      """,
      default: 0.5
    ],
    sequence: [
      doc: """
      If `true` returns a tuple that contains the sequence of intermediate
      results and the index of the last iteration. Otherwise, it returns
      the result of the last iteration.
      """,
      default: false
    ]
  ]

  @solve_opts_schema NimbleOptions.new!(solve_opts_schema)

  @doc """
  Solves the system `AX = B` using a given iterative method.

  ## Options

  #{NimbleOptions.docs(@solve_opts_schema)}
  """
  def solve(a, b, initial_guess, opts) do
    opts = NimbleOptions.validate!(opts, @solve_opts_schema)
    {seq, last_index} = Solvers.iterative_method(a, b, initial_guess, opts)

    if opts[:sequence] do
      {seq, last_index}
    else
      seq[last_index]
    end
  end

  @doc """
  Solves the system `AX = B` directly. It's equivalent to `Nx.LinAlg.solve/2`.
  """
  def solve(a, b), do: Nx.LinAlg.solve(a, b)
end

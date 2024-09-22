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
      type: {:in, [:jacobi, :gauss_seidel, :sor]}
    ],
    omega: [
      doc: "The SOR relaxation factor. Must be a real number between 0 and 2.",
      default: 0.5
    ]
  ]

  @solve_opts_schema NimbleOptions.new!(solve_opts_schema)

  def solve(a, b, initial_guess, opts) do
    opts = NimbleOptions.validate!(opts, @solve_opts_schema)
    Solvers.iterative_method_n(a, b, initial_guess, opts)
  end
end

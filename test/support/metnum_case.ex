defmodule Metnum.Case do
  use ExUnit.CaseTemplate

  using do
    quote do
      import Metnum.TestHelpers
    end
  end
end

# SBArchOpt Interface to SMARTy: Surrogate Modeling for Aero-Data Toolbox

SMARTy is a surrogate modeling toolbox with optimization capabilities developed by the DLR. For more information refer to:

Bekemeyer, P., Bertram, A., Hines Chaves, D.A., Dias Ribeiro, M., Garbo, A., Kiener, A., Sabater, C., Stradtner, M.,
Wassing, S., Widhalm, M. and Goertz, S., 2022. Data-Driven Aerodynamic Modeling Using the DLR SMARTy Toolbox.
In AIAA Aviation 2022 Forum (p. 3899). https://arc.aiaa.org/doi/abs/10.2514/6.2022-3899

## Installation

SMARTy is not openly available.

## Usage

The `get_smarty_optimizer` function can be used to get an interface object for running the optimization.

```python
from sb_arch_opt.algo.smarty_interface import get_smarty_optimizer

problem = ...  # Subclass of ArchOptProblemBase

# Get the interface and optimization loop
smarty = get_smarty_optimizer(problem, n_init=100, n_infill=50)

# Run the optimization loop
smarty.optimize()

# Extract data as a pymoo Population object
pop = smarty.pop
```

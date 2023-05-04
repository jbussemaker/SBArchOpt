# SBArchOpt: Overview of Test Problems

An overview of available test problems in `sb_arch_opt.problems` (* = recommended; further nomenclature is explained below the table):

| Problem Class                     | n_discr | n_cont | n_obj | n_con | D   | MD  | MO  | HIER | HC  | n_valid_discr | imp_ratio | dependencies | Notes                            |
|-----------------------------------|---------|--------|-------|-------|-----|-----|-----|------|-----|---------------|-----------|--------------|----------------------------------|
| Module: `continuous`              |
| `Himmelblau`                      |         | 2      | 1     |       |     |     |     |      |     |               |           |              |                                  |
| `Goldstein`                       |         | 2      | 1     |       |     |     |     |      |     |               |           |              |                                  |
| `Branin`*                         |         | 2      | 1     |       |     |     |     |      |     |               |           |              |                                  |
| `Rosenbrock`*                     |         | 10     | 1     |       |     |     |     |      |     |               |           |              |                                  |
| `Griewank`                        |         | 10     | 1     |       |     |     |     |      |     |               |           |              |                                  |
| Module: `discrete`                |
| `MDBranin`*                       | 2       | 2      | 1     |       |     | Y   |     |      |     | 4             |           |              |                                  |
| `AugmentedMDBranin`*              | 2       | 10     | 1     |       |     | Y   |     |      |     | 4             |           |              |                                  |
| `MDGoldstein`                     | 2       | 2      | 1     |       |     | Y   |     |      |     | 9             |           |              |                                  |
| `MunozZunigaToy`*                 | 1       | 1      | 1     |       |     | Y   |     |      |     | 10            |           |              |                                  |
| `Halstrup04`                      | 3       | 5      | 1     |       |     | Y   |     |      |     | 12            |           |              |                                  |
| Module: `md_mo`                   |
| `MOHimmelblau`                    |         | 2      | 2     |       |     |     | Y   |      |     |               |           |              |                                  |
| `MDMOHimmelblau`                  | 1       | 1      | 2     |       |     | Y   | Y   |      |     | 10            |           |              |                                  |
| `DMOHimmelblau`                   | 2       |        | 2     |       | Y   |     | Y   |      |     | 100           |           |              |                                  |
| `MOGoldstein`                     |         | 2      | 2     |       |     |     | Y   |      |     |               |           |              |                                  |
| `MDMOGoldstein`*                  | 1       | 1      | 2     |       |     | Y   | Y   |      |     | 10            |           |              |                                  |
| `DMOGoldstein`*                   | 2       |        | 2     |       | Y   |     | Y   |      |     | 100           |           |              |                                  |
| `MORosenbrock`*                   |         | 10     | 2     |       |     |     | Y   |      |     |               |           |              |                                  |
| `MDMORosenbrock`*                 | 5       | 5      | 2     |       |     | Y   | Y   |      |     | 1025          |           |              |                                  |
| `MOZDT1`                          |         | 30     | 2     |       |     |     | Y   |      |     |               |           |              |                                  |
| `MDZDT1Small`                     | 6       | 6      | 2     |       |     | Y   | Y   |      |     | 729           |           |              |                                  |
| `MDZDT1Mid`                       | 10      | 10     | 2     |       |     | Y   | Y   |      |     | 59049         |           |              |                                  |
| `MDZDT1`                          | 15      | 15     | 2     |       |     | Y   | Y   |      |     | ~30.5e9       |           |              |                                  |
| `DZDT1`                           | 30      |        | 2     |       | Y   |     | Y   |      |     | ~931e18       |           |              |                                  |
| Module: `constrained`             |
| `ConBraninProd`                   |         | 2      | 1     | 1     |     |     |     |      |     |               |           |              |                                  |
| `ConBraninGomez`                  |         | 2      | 1     | 1     |     |     |     |      |     |               |           |              |                                  |
| `ArchCantileveredBeam`            |         | 4      | 1     | 2     |     |     |     |      |     |               |           |              |                                  |
| `MDCantileveredBeam`              | 2       | 2      | 1     | 2     |     | Y   |     |      |     | 100           |           |              |                                  |
| `ArchWeldedBeam`                  |         | 4      | 2     | 4     |     |     | Y   |      |     |               |           |              |                                  |
| `MDWeldedBeam`                    | 2       | 2      | 2     | 4     |     | Y   | Y   |      |     | 100           |           |              |                                  |
| `ArchCarside`                     |         | 7      | 3     | 10    |     |     | Y   |      |     |               |           |              |                                  |
| `MDCarside`                       | 4       | 3      | 3     | 10    |     | Y   | Y   |      |     | 100000        |           |              |                                  |
| `ArchOSY`                         |         | 6      | 2     | 6     |     |     | Y   |      |     |               |           |              |                                  |
| `MDOSY`                           | 3       | 3      | 2     | 6     |     | Y   | Y   |      |     | 1000          |           |              |                                  |
| `MODASCMOP`                       |         | 30     | 3     | 7     |     |     | Y   |      |     |               |           |              |                                  |
| `MDDASCMOP`                       | 15      | 15     | 3     | 7     |     | Y   | Y   |      |     | ~14.3e6       |           |              |                                  |
| Module: `hierarchical`            |
| `ZaeffererHierarchical`           |         | 2      | 1     |       |     |     |     | Y    |     |               |           |              |                                  |
| `Jenatton`                        | 3       | 6      | 1     |       |     | Y   |     | Y    |     | 4             | 2         |              |                                  |
| `NeuralNetwork`                   | 6       | 2      | 1     |       |     | Y   |     | Y    |     | 4644          | 2.51      |              |                                  |
| `HierarchicalGoldstein`           | 6       | 5      | 1     | 1     |     | Y   |     | Y    |     | 288           | 2.25      |              |                                  |
| `MOHierarchicalGoldstein`         | 6       | 5      | 2     | 1     |     | Y   | Y   | Y    |     | 288           | 2.25      |              |                                  |
| `HierBranin`*                     | 8       | 2      | 1     |       |     | Y   |     | Y    |     | 200           | 3.2       |              |                                  |
| `HierZDT1Small`*                  | 2       | 2      | 2     |       |     | Y   | Y   | Y    |     | 10            | 1.8       |              |                                  |
| `HierZDT1`*                       | 8       | 4      | 2     |       |     | Y   | Y   | Y    |     | 200           | 4.9       |              |                                  |
| `HierZDT1Large`*                  | 9       | 9      | 2     |       |     | Y   | Y   | Y    |     | 2000          | 8.2       |              |                                  |
| `HierDiscreteZDT1`*               | 8       |        | 2     |       | Y   |     | Y   | Y    |     | 2000          | 4.1       |              |                                  |
| `HierCantileveredBeam`*           | 6       | 4      | 1     | 2     |     | Y   |     | Y    |     | 20            | 5.4       |              |                                  |
| `HierCarside`*                    | 6       | 6      | 3     | 10    |     | Y   | Y   | Y    |     | 50            | 6.5       |              |                                  |
| `HierarchicalRosenbrock`          | 5       | 8      | 1     | 2     |     | Y   |     | Y    |     | 32            | 1.5       |              |                                  |
| `MOHierarchicalRosenbrock`        | 5       | 8      | 2     | 2     |     | Y   | Y   | Y    |     | 32            | 1.5       |              |                                  |
| ~~`MOHierarchicalTestProblem`~~   | 11      | 16     | 2     | 2     |     | Y   | Y   | Y    |     | 64            | 72        |              |                                  |
| Module: `gnc`                     |
| `GNCNoActNrType`                  | 18      |        | 1     |       | Y   |     |     | Y    |     | 265           | 989       |              |                                  |
| `GNCNoActType`                    | 20      |        | 2     |       | Y   |     | Y   | Y    |     | 327           | 7.2e3     |              |                                  |
| `GNCNoActNr`                      | 24      |        | 2     |       | Y   |     | Y   | Y    |     | 26500         | 7.2e3     |              |                                  |
| `GNCNoAct`                        | 26      |        | 2     |       | Y   |     | Y   | Y    |     | 29857         | 57.6e3    |              |                                  |
| `GNCNoNrType`                     | 27      |        | 1     |       | Y   |     |     | Y    |     | 70225         | 1911      |              |                                  |
| `GNCNoType`                       | 30      |        | 2     |       | Y   |     | Y   | Y    |     | 85779         | 42.2e3    |              |                                  |
| `GNCNoNr`                         | 36      |        | 2     |       | Y   |     | Y   | Y    |     | 70225000      | 37.6e3    |              |                                  |
| `GNC`                             | 39      |        | 2     |       | Y   |     | Y   | Y    |     | 79091323      | 901e3     |              |                                  |
| Module: `hidden_constraints`      |
| `Mueller01`                       |         | 5      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 67%                   |
| `Mueller02`                       |         | 4      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 40%                   |
| `MDMueller02`                     | 2       | 2      | 1     |       |     | Y   |     |      | Y   | 36            |           |              | fail_rate: 37%                   |
| `Mueller08`                       |         | 10     | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 73%                   |
| `MOMueller08`                     |         | 10     | 2     |       |     |     | Y   |      | Y   |               |           |              | fail_rate: 73%                   |
| `MDMueller08`                     | 2       | 8      | 1     |       |     | Y   |     |      | Y   | 100           |           |              | fail_rate: 78%                   |
| `MDMOMueller08`                   | 2       | 8      | 2     |       |     | Y   | Y   |      | Y   | 100           |           |              | fail_rate: 78%                   |
| `Alimo`                           |         | 2      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 51%                   |
| `AlimoEdge`                       |         | 2      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 53%                   |
| `HCBranin`                        |         | 2      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 33%                   |
| `HCSphere`                        |         | 2      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 51%                   |
| `Tfaily01`                        |         | 2      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 39%                   |
| `Tfaily02`                        |         | 2      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 80%                   |
| `Tfaily03`                        |         | 4      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 67%                   |
| `Tfaily04`                        |         | 6      | 1     |       |     |     |     |      | Y   |               |           |              | fail_rate: 35%                   |
| `CantileveredBeamHC`              |         | 4      | 1     | 1     |     |     |     |      | Y   |               |           |              | fail_rate: 83%                   |
| `MDCantileveredBeamHC`            | 2       | 2      | 1     | 1     |     | Y   |     |      | Y   | 100           |           |              | fail_rate: 81%                   |
| `CarsideHC`                       |         | 7      | 3     | 8     |     |     | Y   |      | Y   |               |           |              | fail_rate: 66%                   |
| `CarsideHCLess`                   |         | 7      | 3     | 9     |     |     | Y   |      | Y   |               |           |              | fail_rate: 39%                   |
| `MDCarsideHC`                     | 4       | 3      | 3     | 8     |     | Y   | Y   |      | Y   | 10000         |           |              | fail_rate: 66%                   |
| `HierAlimo`                       | 5       | 2      | 1     |       |     | Y   |     | Y    | Y   | 20            | 5.4       |              | fail_rate: 51%                   |
| `HierAlimoEdge`                   | 5       | 2      | 1     |       |     | Y   |     | Y    | Y   | 20            | 5.4       |              | fail_rate: 53%                   |
| `HierMueller02`                   | 4       | 4      | 1     |       |     | Y   |     | Y    | Y   | 20            | 5.4       |              | fail_rate: 37%                   |
| `HierMueller08`                   | 4       | 10     | 1     |       |     | Y   |     | Y    | Y   | 20            | 5.4       |              | fail_rate: 73%                   |
| `MOHierMueller08`                 | 4       | 10     | 2     |       |     | Y   | Y   | Y    | Y   | 20            | 5.4       |              | fail_rate: 73%                   |
| `HierarchicalRosenbrockHC`        | 5       | 8      | 1     | 1     |     | Y   | Y   | Y    | Y   | 32            | 1.5       |              | fail_rate: 21%                   |
| `MOHierarchicalRosenbrockHC`      | 5       | 8      | 2     | 1     |     | Y   | Y   | Y    | Y   | 32            | 1.5       |              | fail_rate: 60%                   |
| ~~`HCMOHierarchicalTestProblem`~~ | 11      | 16     | 2     | 1     |     | Y   | Y   | Y    | Y   | 64            | 72        |              | fail_rate: 60%                   |
| Module: `turbofan_arch`           |         |        |       |       |     |     |     |      |     |               |           | `ota`        |                                  |
| `SimpleTurbofanArch`              | 6       | 9      | 1     | 5     |     | Y   |     | Y    | Y   | 70            | 3.1       | `ota`        | t_eval: 1-5 min; fail_rate: 51%  |
| `RealisticTurbofanArch`           | 11      | 30     | 3     | 15    |     | Y   | Y   | Y    | Y   | 142243        | 9.1       | `ota`        | t_eval: 1-5 min; fail_rate: 67%  |
| Module: `assignment`              |         |        |       |       |     |     |     |      |     |               |           | `assignment` | dist_corr: 100% if not specified |
| `Assignment`                      | 12      |        | 2     |       | Y   |     | Y   |      |     | 4096          |           | `assignment` |                                  |
| `AssignmentLarge`                 | 16      |        | 2     |       | Y   |     | Y   |      |     | 65536         |           | `assignment` |                                  |
| `AssignmentInj`                   | 5       |        | 2     |       | Y   |     | Y   |      |     | 7776          |           | `assignment` | dist_corr: 31%                   |
| `AssignmentInjLarge`              | 18      |        | 2     |       | Y   |     | Y   |      |     | 117649        |           | `assignment` | dist_corr: 31%                   |
| `AssignmentRepeat`                | 8       |        | 2     |       | Y   |     | Y   |      |     | 6561          |           | `assignment` |                                  |
| `AssignmentRepeatLarge`           | 10      |        | 2     |       | Y   |     | Y   |      |     | 59049         |           | `assignment` |                                  |
| `Partitioning`                    | 12      |        | 2     |       | Y   |     | Y   |      |     | 4096          |           | `assignment` | dist_corr: 74%                   |
| `PartitioningLarge`               | 7       |        | 2     |       | Y   |     | Y   |      |     | 78125         |           | `assignment` | dist_corr: 67%                   |
| `PartitioningCovering`            | 12      |        | 2     |       | Y   |     | Y   | Y    |     | 2401          | 1.71      | `assignment` |                                  |
| `PartitioningCoveringLarge`       | 16      |        | 2     |       | Y   |     | Y   | Y    |     | 50625         | 1.29      | `assignment` |                                  |
| `Downselecting`                   | 12      |        | 2     |       | Y   |     | Y   |      |     | 4096          |           | `assignment` |                                  |
| `DownselectingLarge`              | 15      |        | 2     |       | Y   |     | Y   |      |     | 32768         |           | `assignment` |                                  |
| `Connecting`                      | 12      |        | 2     |       | Y   |     | Y   |      |     | 4096          |           | `assignment` |                                  |
| `ConnectingUndirected`            | 15      |        | 2     |       | Y   |     | Y   |      |     | 32768         |           | `assignment` |                                  |
| `Permuting`                       | 6       |        | 2     |       | Y   |     | Y   |      |     | 5040          |           | `assignment` | dist_corr: 43%                   |
| `PermutingLarge`                  | 7       |        | 2     |       | Y   |     | Y   |      |     | 40320         |           | `assignment` | dist_corr: 39%                   |
| `UnordNonReplComb`                | 14      |        | 2     |       | Y   |     | Y   | Y    |     | 6435          | 2.55      | `assignment` |                                  |
| `UnordNonReplCombLarge`           | 18      |        | 2     |       | Y   |     | Y   | Y    |     | 92378         | 2.84      | `assignment` |                                  |
| `UnorderedComb`                   | 1       |        | 2     |       | Y   |     | Y   |      |     | 2002          |           | `assignment` | dist_corr: 34%                   |
| `AssignmentGNCNoActType`          | 11      |        | 2     |       | Y   |     | Y   | Y    |     | 327           | 14.1      | `assignment` |                                  |
| `AssignmentGNCNoAct`              | 15      |        | 2     |       | Y   |     | Y   | Y    |     | 29857         | 39.5      | `assignment` |                                  |
| `AssignmentGNCNoType`             | 21      |        | 2     |       | Y   |     | Y   | Y    |     | 85779         | 82.5      | `assignment` |                                  |
| `AssignmentGNC`                   | 27      |        | 2     |       | Y   |     | Y   | Y    |     | 79091323      | 367       | `assignment` |                                  |

Nomenclature:
- n_discr: number of discrete (integer or categorical) design variables
- n_cont: number of continuous design variables
- n_obj: number of objectives
- n_con: number of (inequality) constraints
- D: whether the problem contains discrete design variables
- MD: whether the problem is a mixed-discrete problem or not
- MO: whether the problem is a multi-objective problem or not
- HIER: whether the problem contains hierarchical variables
- HC: whether the problem contains hidden constraints (i.e. some points might fail to evaluate); see also failure_rate
- n_valid_discr: number of valid discrete design points (i.e. ignoring continuous dimensions)
- imp_ratio: imputation ratio; ratio between the number of declared and valid discrete design points (1 means there are
  no invalid design vectors)
- dependencies: name of the optional dependencies list to install to use the test problem (`pip install -e .[name]`)
- fail_rate: fraction of randomly-sampled points that fail to evaluate
- t_eval: rough estimate of the time it takes to evaluate one design point (practically instantaneous if left empty)
- dist_corr: distance correlation between design vectors and assignment patterns (higher is better)

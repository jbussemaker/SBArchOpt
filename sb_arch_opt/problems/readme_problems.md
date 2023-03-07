# SBArchOpt: Overview of Test Problems

An overview of available test problems in `sb_arch_opt.problems`:

| Problem Class                 | n_discr | n_cont | n_obj | n_con | D   | MD  | MO  | HIER | HC  | n_valid_discr | imp_ratio | t_eval  | dependencies |
|-------------------------------|---------|--------|-------|-------|-----|-----|-----|------|-----|---------------|-----------|---------|--------------|
| `MDBranin`                    | 2       | 2      | 1     |       |     | Y   |     |      |     | 4             |           |         |              |
| `AugmentedMDBranin`           | 2       | 10     | 1     |       |     | Y   |     |      |     | 4             |           |         |              |
| `MDGoldstein`                 | 2       | 2      | 1     |       |     | Y   |     |      |     | 9             |           |         |              |
| `MunozZunigaToy`              | 1       | 1      | 1     |       |     | Y   |     |      |     | 10            |           |         |              |
| `Halstrup04`                  | 3       | 5      | 1     |       |     | Y   |     |      |     | 12            |           |         |              |
| `MOHimmelblau`                |         | 2      | 2     |       |     |     | Y   |      |     |               |           |         |              |
| `MDMOHimmelblau`              | 1       | 1      | 2     |       |     | Y   | Y   |      |     | 10            |           |         |              |
| `DMOHimmelblau`               | 2       |        | 2     |       | Y   |     | Y   |      |     | 100           |           |         |              |
| `MOGoldstein`                 |         | 2      | 2     |       |     |     | Y   |      |     |               |           |         |              |
| `MDMOGoldstein`               | 1       | 1      | 2     |       |     | Y   | Y   |      |     | 10            |           |         |              |
| `DMOGoldstein`                | 2       |        | 2     |       | Y   |     | Y   |      |     | 100           |           |         |              |
| `MOZDT1`                      |         | 30     | 2     |       |     |     | Y   |      |     |               |           |         |              |
| `MDZDT1`                      | 15      | 15     | 2     |       |     | Y   | Y   |      |     | ~30.5e9       |           |         |              |
| `DZDT1`                       | 30      |        | 2     |       | Y   |     | Y   |      |     | ~931e18       |           |         |              |
| `ArchWeldedBeam`              |         | 4      | 2     | 4     |     |     | Y   |      |     |               |           |         |              |
| `MDWeldedBeam`                | 2       | 2      | 2     | 4     |     | Y   | Y   |      |     | 100           |           |         |              |
| `ArchCarside`                 |         | 7      | 3     | 10    |     |     | Y   |      |     |               |           |         |              |
| `MDCarside`                   | 4       | 3      | 3     | 10    |     | Y   | Y   |      |     | 100000        |           |         |              |
| `ArchOSY`                     |         | 6      | 2     | 6     |     |     | Y   |      |     |               |           |         |              |
| `MDOSY`                       | 3       | 3      | 2     | 6     |     | Y   | Y   |      |     | 1000          |           |         |              |
| `MODASCMOP`                   |         | 30     | 3     | 7     |     |     | Y   |      |     |               |           |         |              |
| `MDDASCMOP`                   | 15      | 15     | 3     | 7     |     | Y   | Y   |      |     | ~14.3e6       |           |         |              |
| `ZaeffererHierarchical`       |         | 2      | 1     |       |     |     |     | Y    |     |               |           |         |              |
| `HierarchicalGoldstein`       | 6       | 5      | 1     | 1     |     | Y   |     | Y    |     | 288           | 2.25      |         |              |
| `MOHierarchicalGoldstein`     | 6       | 5      | 2     | 1     |     | Y   | Y   | Y    |     | 288           | 2.25      |         |              |
| `HierarchicalRosenbrock`      | 5       | 8      | 1     | 2     |     | Y   |     | Y    |     | 32            | 1.5       |         |              |
| `MOHierarchicalRosenbrock`    | 5       | 8      | 2     | 2     |     | Y   | Y   | Y    |     | 32            | 1.5       |         |              |
| `MOHierarchicalTestProblem`   | 11      | 16     | 2     | 2     |     | Y   | Y   | Y    |     | 64            | 72        |         |              |
| `MOHierarchicalRosenbrockHC`  | 5       | 8      | 2     | 1     |     | Y   | Y   | Y    | Y   | 32            | 1.5       |         |              |
| `HCMOHierarchicalTestProblem` | 11      | 16     | 2     | 1     |     | Y   | Y   | Y    | Y   | 64            | 72        |         |              |
| `GNCNoActNrType`              | 18      |        | 1     |       | Y   |     |     | Y    |     | 265           | 989       |         |              |
| `GNCNoActType`                | 20      |        | 2     |       | Y   |     | Y   | Y    |     | 327           | 7.2e3     |         |              |
| `GNCNoActNr`                  | 24      |        | 2     |       | Y   |     | Y   | Y    |     | 26500         | 7.2e3     |         |              |
| `GNCNoAct`                    | 26      |        | 2     |       | Y   |     | Y   | Y    |     | 29857         | 57.6e3    |         |              |
| `GNCNoNrType`                 | 27      |        | 1     |       | Y   |     |     | Y    |     | 70225         | 1911      |         |              |
| `GNCNoType`                   | 30      |        | 2     |       | Y   |     | Y   | Y    |     | 85779         | 42.2e3    |         |              |
| `GNCNoNr`                     | 36      |        | 2     |       | Y   |     | Y   | Y    |     | 70225000      | 37.6e3    |         |              |
| `GNC`                         | 39      |        | 2     |       | Y   |     | Y   | Y    |     | 79091323      | 901e3     |         |              |
| `SimpleTurbofanArch`          | 6       | 9      | 1     | 5     |     | Y   |     | Y    | Y   | 70            | 3.1       | 1-5 min | ota          |
| `RealisticTurbofanArch`       | 11      | 30     | 3     | 15    |     | Y   | Y   | Y    | Y   | 1163          | 1114      | 1-5 min | ota          |

Nomenclature:
- n_discr: number of discrete (integer or categorical) design variables
- n_cont: number of continuous design variables
- n_obj: number of objectives
- n_con: number of (inequality) constraints
- D: whether the problem contains discrete design variables
- MD: whether the problem is a mixed-discrete problem or not
- MO: whether the problem is a multi-objective problem or not
- HIER: whether the problem contains hierarchical variables
- HC: whether the problem contains hidden constraints (i.e. some points might fail)
- n_valid_discr: number of valid discrete design points (i.e. ignoring continuous dimensions)
- imp_ratio: imputation ratio; ratio between the number of declared and valid discrete design points (1 means there are
  no invalid design vectors)
- t_eval: rough estimate of the time it takes to evaluate one design point (practically instantaneous if left empty)
- dependencies: name of the optional dependencies list to install to use the test problem (`python setup.py install[name]`)

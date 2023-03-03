# SBArchOpt: Overview of Test Problems

An overview of available test problems in `sb_arch_opt.problems`:

| Problem Class    | n_discr | n_cont | n_obj | n_con | D   | MD  | MO  | HIER | HC  | n_valid_discr | imp_ratio | t_eval |
|------------------|---------|--------|-------|-------|-----|-----|-----|------|-----|---------------|-----------|--------|
| `MOHimmelblau`   |         | 2      | 2     |       |     |     | Y   |      |     |               |           |        |
| `MDMOHimmelblau` | 1       | 1      | 2     |       | Y   | Y   | Y   |      |     | 10            |           |        |
| `DMOHimmelblau`  | 2       |        | 2     |       | Y   |     | Y   |      |     | 100           |           |        |
| `MOGoldstein`    |         | 2      | 2     |       |     |     | Y   |      |     |               |           |        |
| `MDMOGoldstein`  | 1       | 1      | 2     |       | Y   | Y   | Y   |      |     | 10            |           |        |
| `DMOGoldstein`   | 2       |        | 2     |       | Y   |     | Y   |      |     | 100           |           |        |
| `MOZDT1`         |         | 30     | 2     |       |     |     | Y   |      |     |               |           |        |
| `MDZDT1`         | 15      | 15     | 2     |       | Y   | Y   | Y   |      |     | ~30.5e9       |           |        |
| `DZDT1`          | 30      |        | 2     |       | Y   |     | Y   |      |     | ~931e18       |           |        |

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

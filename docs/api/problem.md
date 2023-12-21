# Problem Definition and Sampling

::: sb_arch_opt.problem.ArchOptProblemBase
    handler: python
    options:
        heading_level: 2
        members:
            - design_space
            - correct_x
            - load_previous_results
            - store_results
            - evaluate
            - vars
            - get_categorical_values
            - is_conditionally_active
            - all_discrete_x
            - print_stats
            - get_imputation_ratio
            - get_correction_ratio
            - get_discrete_rates
            - get_failure_rate
            - get_n_declared_discrete
            - get_n_valid_discrete
            - get_n_correct_discrete
            - _arch_evaluate
            - _correct_x

::: sb_arch_opt.sampling.HierarchicalSampling
    handler: python
    options:
        heading_level: 2
        members:
            - sample_get_x

::: sb_arch_opt.design_space.ArchDesignSpace
    handler: python
    options:
        heading_level: 2
        members:
            - all_discrete_x
            - correct_x
            - corrector
            - quick_sample_discrete_x
            - des_vars
            - imputation_ratio
            - correction_ratio
            - is_conditionally_active
            - is_cat_mask
            - is_cont_mask
            - is_discrete_mask
            - is_int_mask
            - xl
            - xu
            - get_categorical_values
            - get_discrete_rates
            - get_n_declared_discrete
            - get_n_valid_discrete
            - get_n_correct_discrete

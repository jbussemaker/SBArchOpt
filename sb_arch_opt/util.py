import logging.config

_debug_log_captured = False


def capture_log(level='INFO'):
    """Displays logging output in the console"""
    global _debug_log_captured
    if _debug_log_captured:
        return
    if level == 'DEBUG':
        _debug_log_captured = True

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'console': {
                'format': '%(levelname)- 8s %(asctime)s %(name)- 18s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'console',
            },
        },
        'loggers': {
            'sb_arch_opt': {
                'handlers': ['console'],
                'level': level,
            },
        },
    })


def patch_ftol_bug(term):  # Already fixed in upcoming release: https://github.com/anyoptimization/pymoo/issues/325
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    from pymoo.termination.ftol import MultiObjectiveSpaceTermination
    data_func = None

    def _wrap_data(algorithm):
        data = data_func(algorithm)
        if data['ideal'] is None:
            data['feas'] = False
        return data

    if isinstance(term, DefaultMultiObjectiveTermination):
        ftol_term = term.criteria[2].termination
        if isinstance(ftol_term, MultiObjectiveSpaceTermination):
            data_func = ftol_term._data
            ftol_term._data = _wrap_data

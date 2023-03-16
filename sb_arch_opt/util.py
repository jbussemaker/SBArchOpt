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

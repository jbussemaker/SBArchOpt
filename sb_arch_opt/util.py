import logging.config


def capture_log(level='INFO'):
    """Displays logging output in the console"""
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

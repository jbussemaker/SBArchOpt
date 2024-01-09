import os
import logging.config
from appdirs import user_cache_dir

_debug_log_captured = False


def _prevent_capture():
    global _debug_log_captured
    _debug_log_captured = True


def capture_log(level='INFO'):
    """Displays logging output in the console"""
    global _debug_log_captured
    if _debug_log_captured:
        return
    if level == 'DEBUG':
        _debug_log_captured = True

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
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


def get_np_random_singleton():
    from scipy._lib._util import check_random_state
    return check_random_state(seed=None)


def get_cache_path(sub_path: str = None):
    path = user_cache_dir('SBArchOpt')
    os.makedirs(path, exist_ok=True)
    if sub_path is not None:
        path = os.path.join(path, sub_path)
    return path

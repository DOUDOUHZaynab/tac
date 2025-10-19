import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str = __name__, level: int = logging.INFO):
    """Return a configured logger for the project."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (rotating)
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'logs')
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(os.path.join(log_dir, 'tp1.log'), maxBytes=5_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # if file handler can't be created, continue with console only
        pass

    return logger

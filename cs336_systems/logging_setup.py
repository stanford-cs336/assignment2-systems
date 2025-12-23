# logging_setup.py
import logging, logging.handlers, os
from datetime import datetime


def setup_logger(
    run_name="lm_run",
    log_dir="artifacts/logs",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = os.path.join(log_dir, f"{run_name}-{ts}.log")

    log = logging.getLogger(run_name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()  # <--- important
    log.propagate = False  # <--- important: don’t bubble to root

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=50_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(file_level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)

    return log, logfile

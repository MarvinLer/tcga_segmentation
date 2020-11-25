__author__ = 'marvinler'

import os
import logging

# max number of processes for parallelism
N_PROCESSES = 5


def get_logger(filename_handler, verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # file handler
        filepath = os.path.join('logs', filename_handler)
        if not os.path.exists(os.path.dirname(os.path.abspath(filepath))):
            os.makedirs(os.path.dirname(os.path.abspath(filepath)))
        fh = logging.FileHandler(filepath)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%d/%m/%Y %I:%M:%S %p')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter('%(levelname)s     %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

__author__ = 'marvinler'

import logging
import sys

# max number of processes for parallelism
N_PROCESSES = 5


def get_logger(filename_handler, verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(filename_handler)
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

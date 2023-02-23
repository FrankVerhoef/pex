## This code is an adjusted version of logging.py from Parlai

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from datetime import datetime

try:
    import coloredlogs
    COLORED_LOGS = True
except ImportError:
    COLORED_LOGS = False

# Name of the logger
LOGGER_NAME = "pex"

# Define log levels
SPAM = logging.DEBUG - 5
DEBUG = logging.DEBUG
VERBOSE = DEBUG + 5
INFO = logging.INFO
REPORT = INFO + 5
SUCCESS = REPORT + 1
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(SPAM, "SPAM")
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(REPORT, "REPORT")
logging.addLevelName(SUCCESS, "SUCCESS")

# Define formats and colors for logging messages
MESSAGE_FORMAT = '%(message)s'
COLORED_FORMAT = '%(asctime)s | %(message)s'
CONSOLE_FORMAT = '%(asctime)s %(levelname)-8s | %(message)s'
CONSOLE_DATE_FORMAT = '%H:%M:%S'
LOGFILE_FORMAT = '%(asctime)s %(levelname)-8s | %(message)s'
LOGFILE_DATE_FORMAT = None

COLORED_LEVEL_STYLES = {
    'spam': {'color': 'white', 'faint': True},
    'debug': {'color': 'blue'},
    'verbose': {'color': 'blue', 'faint': True},
    'info': {},
    'report': {'bold': True},
    'success': {'bold': True, 'color': 'green'},
    'warning': {'color': 'yellow'},
    'error': {'color': 'red'},
    'critical': {'bold': True, 'color': 'red'},
}


class PexLogger(logging.Logger):
    def __init__(self, name, console_level=INFO):
        """
        Initialize the logger object.

        :param name:
            Name of the logger
        :param console_level:
            minimum level of messages logged to console
        """

        super().__init__(name, console_level)
        self.onlymessage = False
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.streamHandler.setFormatter(self._build_formatter())
        super().addHandler(self.streamHandler)


    def _build_formatter(self):
        if COLORED_LOGS and sys.stdout.isatty():
            return coloredlogs.ColoredFormatter(
                MESSAGE_FORMAT if self.onlymessage else COLORED_FORMAT,
                datefmt=CONSOLE_DATE_FORMAT,
                level_styles=COLORED_LEVEL_STYLES,
                field_styles={},
            )
        elif sys.stdout.isatty():
            return logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
        else:
            return logging.Formatter(LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT)

    def set_only_message(self, value):
        self.onlymessage = value
        self.streamHandler.setFormatter(self._build_formatter())    

    def add_file_handler(self, logdir):
        self.fileHandler = logging.FileHandler(logdir + LOGGER_NAME + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
        self.fileHandler.setFormatter(logging.Formatter(LOGFILE_FORMAT, datefmt=LOGFILE_DATE_FORMAT))
        super().addHandler(self.fileHandler)

    def log(self, msg, level=INFO):
        """
        Default Logging function.
        """
        super().log(level, msg)

    def mute(self):
        """
        Stop logging to stdout.
        """
        self.prev_level = self.streamHandler.level
        self.streamHandler.level = ERROR
        return self.prev_level

    def unmute(self):
        """
        Resume logging to stdout.
        """
        self.streamHandler.level = self.prev_level


# Initialize the logger
logger = PexLogger(name=LOGGER_NAME)

def set_log_level(level):
    logger.setLevel(level)

def set_only_message(value):
    logger.set_only_message(value)

def add_file_handler(logdir):
    logger.add_file_handler(logdir)

def disable():
    logger.mute()

def enable():
    logger.unmute()

def log(*args, **kwargs):
    return logger.log(*args, **kwargs)

def spam(msg):
    return logger.log(msg, level=SPAM)

def debug(*args, **kwargs):
    return logger.debug(*args, **kwargs)

def verbose(msg):
    return logger.log(msg, level=VERBOSE)

def info(msg):
    return logger.info(msg)

def report(msg):
    return logger.log(msg, level=REPORT)

def success(msg):
    return logger.log(msg, level=SUCCESS)

def warning(*args, **kwargs):
    return logger.warning(*args, **kwargs)

def error(*args, **kwargs):
    return logger.error(*args, **kwargs)

def critical(msg):
    return logger.critical(msg)

def get_all_levels():
    levels = set(logging._nameToLevel.keys())
    levels.remove('WARN')
    return [l for l in levels]
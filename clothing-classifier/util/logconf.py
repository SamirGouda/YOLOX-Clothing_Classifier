#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

datefmt='%H:%M:%S'
format="%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(format, datefmt)

logging.basicConfig(
        level=logging.INFO, datefmt=datefmt, format=format,
        handlers=[
            logging.StreamHandler()
        ]
        )


# def get_logger(name, log_path):
    # logging.basicConfig(
    #     level=logging.INFO, datefmt='%H:%M:%S',
    #     format="%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s",
    #     handlers=[
    #         logging.FileHandler(log_path),
    #         logging.StreamHandler()
    #     ]
    #     )
    # root_logger = logging.getLogger(name)
    # root_logger = logging.getLogger(name)
    # root_logger.setLevel(logging.INFO)

    # # remove libraries root logger handlers
    # for handler in list(root_logger.handlers):
    #     root_logger.removeHandler(handler)



    # logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
    # formatter = logging.Formatter(logfmt_str)

    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(formatter)
    # streamHandler.setLevel(logging.DEBUG)

    # root_logger.addHandler(streamHandler)

    # fileHandler = logging.FileHandler(log_path)
    # fileHandler.setFormatter(formatter)
    # fileHandler.setLevel(logging.DEBUG)

    # root_logger.addHandler(fileHandler)

    # return root_logger

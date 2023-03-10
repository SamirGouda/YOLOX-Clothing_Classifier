#! /usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import time
from tqdm import tqdm
from time import sleep
from pathlib import Path
from util.logconf import logging
import yaml
# from hyperpyyaml import load_hyperpyyaml

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def enumerateWithEstimate(iter_, desc_str, start_ndx=0, print_ndx=4, backoff=None, iter_len=None):
    """
        In terms of behavior, `enumerateWithEstimate` is almost identical
        to the standard `enumerate` (the differences are things like how
        our function returns a generator, while `enumerate` returns a
        specialized `<enumerate object at 0x...>`).

        However, the side effects (logging, specifically) are what make the
        function interesting.

        :param iter_: `iter_` is the iterable that will be passed into
            `enumerate`. Required.

        :param desc_str: This is a human-readable string that describes
            what the loop is doing. The value is arbitrary, but should be
            kept reasonably short. Things like `"epoch 4 training"` or
            `"deleting temp files"` or similar would all make sense.

        :param start_ndx: This parameter defines how many iterations of the
            loop should be skipped before timing actually starts. Skipping
            a few iterations can be useful if there are startup costs like
            caching that are only paid early on, resulting in a skewed
            average when those early iterations dominate the average time
            per iteration.

            NOTE: Using `start_ndx` to skip some iterations makes the time
            spent performing those iterations not be included in the
            displayed duration. Please account for this if you use the
            displayed duration for anything formal.

            This parameter defaults to `0`.

        :param print_ndx: determines which loop iteration that the timing
            logging will start on. The intent is that we don't start
            logging until we've given the loop a few iterations to let the
            average time-per-iteration a chance to stablize a bit. We
            require that `print_ndx` not be less than `start_ndx` times
            `backoff`, since `start_ndx` greater than `0` implies that the
            early N iterations are unstable from a timing perspective.

            `print_ndx` defaults to `4`.

        :param backoff: This is used to how many iterations to skip before
            logging again. Frequent logging is less interesting later on,
            so by default we double the gap between logging messages each
            time after the first.

            `backoff` defaults to `2` unless iter_len is > 1000, in which
            case it defaults to `4`.

        :param iter_len: Since we need to know the number of items to
            estimate when the loop will finish, that can be provided by
            passing in a value for `iter_len`. If a value isn't provided,
            then it will be set by using the value of `len(iter_)`.

        :return:
        """
    if iter_len is None:
        iter_len = len(iter_)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    logging.warning("{} ----/{}, starting".format(desc_str, iter_len))
    start_ts = time.time()
    for current_ndx, item in enumerate(iter_):
        yield current_ndx, item
        if current_ndx == print_ndx:
            duration_sec = ((time.time() - start_ts) / (current_ndx - start_ndx + 1) *
                            (iter_len - start_ndx))
            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)
            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str, current_ndx, iter_len,
                str(done_dt).rsplit('.', 1)[0], str(done_td).rsplit('.', 1)[0],
            ))
            print_ndx *= backoff
        if current_ndx + 1 == start_ndx:
            start_ts = time.time()
    log.warning("{} ----/{}, done at {}".format(
        desc_str, iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

format_ = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{unit}]'

def enumerateWithEstimate1(iter_, desc_str, start_ndx=0, print_ndx=4, backoff=None, iter_len=None):
    if iter_len is None:
        iter_len = len(iter_)
    with tqdm(iterable=iter_, desc=desc_str, total=iter_len, unit='', bar_format=format_) as pbar:
        for current_ndx, item in enumerate(iter_):
            yield current_ndx, item
            sleep(0.1)
            pbar.update(100)

def read_classes_file(file: Path, value_fn = int):
    tmp_list = []
    tmp_dict = {}
    with open(file, 'r') as fd:
        for line in fd:
            class_, class_id = line.strip().split()
            tmp_dict[class_] = value_fn(class_id)
            tmp_list.append(class_)
    
    return tmp_dict, tmp_list

def read_yaml_conf(file: Path):
    with open(file, 'r') as fd:
        conf = yaml.load(fd, Loader=yaml.FullLoader)
    
    return conf
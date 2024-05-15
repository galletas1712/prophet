import torch
import itertools


SCHEDULERS = {}


def register_scheduler(name):
    def register_curr_scheduler(scheduler_class):
        SCHEDULERS[name] = scheduler_class
        return scheduler_class

    return register_curr_scheduler


def build_scheduler(scheduler_config, **kwargs):
    return SCHEDULERS[scheduler_config.name](**scheduler_config, **kwargs)

import torch
import itertools


SCHEDULERS = {}


def register_scheduler(name):
    def register_curr_scheduler(scheduler_class):
        SCHEDULERS[name] = scheduler_class
        return scheduler_class


def create_scheduler(scheduler_config):
    return SCHEDULERS[scheduler_config.name](**scheduler_config)

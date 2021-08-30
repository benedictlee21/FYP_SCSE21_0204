
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group=1000, ratio=1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i >= num_group:
                param_group['lr'] = 0
            else:
                param_group['lr'] = learning_rate * ratio**i


import datetime
import time
from functools import wraps
from typing import Any, Callable


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Times a function, usually used as decorator"""
    # ref: http://zyxue.github.io/2017/09/21/python-timeit-decorator.html
    @wraps(func)
    def timed_func(*args: Any, **kwargs: Any) -> Any:
        """Returns the timed function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
        print("time spent on %s: %s"%(func.__name__, elapsed_time))
        return result

    return timed_func
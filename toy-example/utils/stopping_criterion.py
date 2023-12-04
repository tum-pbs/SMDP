from enum import Enum

stoppingCriterionType = Enum('StoppingCriterionType', 'MAX_GRAD_UPDATES MAX_TIME MAX_EPOCHS')

import datetime


class StoppingCriterion(object):
    """Base class for stopping criterion. """

    def __init__(self, stop_type: stoppingCriterionType, tol):
        self.stop_type = stop_type
        self.tol = tol

        self.start_time = datetime.datetime.now()

    def __call__(self, epoch, grad_updates):

        now = datetime.datetime.now()

        passed_time_minutes = (now - self.start_time).total_seconds() / 60.0

        # iterate of all stopping criterion types
        for criterion in stoppingCriterionType:
            # check if current stopping criterion is of type
            if criterion == self.stop_type:
                # check if stopping criterion is fulfilled
                if criterion == stoppingCriterionType.MAX_GRAD_UPDATES:
                    return grad_updates >= self.tol
                elif criterion == stoppingCriterionType.MAX_TIME:
                    return passed_time_minutes >= self.tol
                elif criterion == stoppingCriterionType.MAX_EPOCHS:
                    return epoch >= self.tol
                else:
                    raise NotImplementedError

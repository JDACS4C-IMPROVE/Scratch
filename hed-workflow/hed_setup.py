
from runner_utils import ModelResult

def pre_run(params):
    print("hed_setup: " + str(params))
    return ModelResult.SUCCESS

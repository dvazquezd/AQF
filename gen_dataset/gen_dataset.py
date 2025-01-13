import os
import sys
import pandas as pd
import utils.utils as ut


def run_gen_dataset():
    '''    
    '''
    #Objects
    config = ut.load_config('gen_dataset_config')
    return config



if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_gen_dataset()
import config

import numpy as np
import pandas as pd
from Neural_Network import *

# run_type = "TRAIN"
# run_type = "TEST ONLY"
best_params = "BEST"

NN = Neural_Network(config.inputLayerSize, config.outputLayerSize, config.hiddenLayerSize)


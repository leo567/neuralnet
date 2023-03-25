import numpy as np

inputLayerSize = 4
outputLayerSize = 1
hiddenLayerSize = 3

# initial weights
initW1 = np.random.randn(inputLayerSize, hiddenLayerSize)
initW2 = np.random.randn(hiddenLayerSize, outputLayerSize)

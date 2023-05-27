import pandas as pd
import numpy as np
from generateDataFunctions import normalizeData
from FCI import FCI
from generateData import generateSingleDataSet

data, graph = generateSingleDataSet(N=400, V=10, latent_rate=.2, linear=False, GP=True)
print(graph.adjacencyMatrix())
pag = FCI(graph.all_variables, data=data, indTest='kernel', CDC=True, aug_PC=True, verbose=True)
print(pag.adjacencyMatrix())
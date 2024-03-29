import numpy as np
from causalGraph import CausalGraph
from generateDataFunctions import noise, normalizeData, coefficient, function, GaussianProcessEffect
import pandas as pd
from searchFunctions import findSepSets
import json
import os
import math
from genMAG import mk_mag
from TarjansAlgorithm import TarjansAlgorithm


def generatePairwiseDataGaussianProcess(N=500, cause=False, noise_type=None, num_confounders=0, num_latent=0, nd = 1, additive_noise=True):
    """
    Returns data generated by a Gaussian process following the causal relationship X1->X3 with X2s as confounders.
        :param N: sample size, default 500
        :param cause: boolean that indicates if there is a causal relation x1->x3
        :param noise_type: noise type ('Gaussian' or 'Uniform')
        :param num_confounders: number of confounders
        :param num_latent: number of hidden confounders
        :param nd: noise depedence factor
        :param additive_noise: boolean indicating if there must be additive noise
    """

    confounders = []
    for i in range(num_confounders+num_latent):
        confounders.append(noise(noise_type, N))

    # generate data of x1
    x1 = GaussianProcessEffect(N, tuple(confounders), noise_type, additive_noise)

    if cause:
        cause_data = tuple(confounders) + (x1,)
    else:
        cause_data = tuple(confounders)

    # generate data of x3
    x3 = GaussianProcessEffect(N, cause_data, noise_type, additive_noise)

    # normalize x1 and x3
    x1 = normalizeData(x1)
    x3 = normalizeData(x3)

    # normalize observed confounders
    if num_confounders>0:
        x2 = normalizeData(confounders[0])
        for i in range(1,num_confounders):
            x2 = np.append(x2,normalizeData(confounders[i]),1)
    else:
        x2 = None

    return x1, x2, x3


def generatePairwiseDataFunctional(N=500, cause=False, noise_type=None, num_confounders=0, num_latent=0, nd = 1, ratio2noise=False, additive_noise=True):
    """
    Returns data generated with randomly chosen functional causal relationships following the causal relationship X1->X3 with X2s as confounders.
        :param N: sample size, default 500
        :param cause: boolean that indicates if there is a causal relation x1->x3
        :param noise_type: noise type ('Gaussian' or 'Uniform')
        :param num_confounders: number of confounders
        :param num_latent: number of hidden confounders
        :param nd: noise depedence factor
        :param ratio2noise: boolean that indicated that we will have a cause with multiple strengths (only possible when cause==True)
        :param additive_noise: boolean indicating if there must be additive noise
    """

    # initialize
    x1, noiseX1 = noise(noise_type, N, track=True)
    x1 = nd * x1
    x2 = None
    x3, noiseX3 = noise(noise_type, N, track=True)
    x3 = nd * x3
    x3s = None

    # add confounding effect
    confounders = []
    for i in range(num_confounders+num_latent):
        confounders.append(noise(noise_type, N))
        x1 += function(confounders[-1], noise_type=noiseX1)
        x3 += function(confounders[-1], noise_type=noiseX3)

    # add cause x1->x3 on strength
    if cause and ratio2noise:
        # none = almost no cause,..., extreme= almost no noise
        strengths = {'none':0.01,'weak':.25, 'neutral':.5, 'strong':.75, 'extreme':.99}
        x3s = {}
        for strength in strengths.keys():
            x3s[strength] = (1-strengths[strength])*x3 + strengths[strength]*function(x1, noise_type=noiseX3)
    elif cause:
        x3 += function(x1, noise_type=noiseX3) + x3

    # normalize x1
    x1 = normalizeData(x1)

    # normalize observed confounders
    if num_confounders>0:
        x2 = normalizeData(confounders[0])
        for i in range(1,num_confounders):
            x2 = np.append(x2,normalizeData(confounders[i]),1)

    # normalize x3 and return
    if x3s is not None:
        for strength in strengths:
            x3s[strength] = normalizeData(x3s[strength])
        return x1, x2, x3s
    elif x3 is not None:
        x3 = normalizeData(x3)
        return x1, x2, x3


def generatePairwiseDataLinearGaussian(N=500, cause=False, num_confounders=0, num_latent=0, nd = 1, ratio2noise=False, additive_noise=True):
    """
    Returns linear Gaussian data following the causal relationship X1->X3 with X2s as confounders.
        :param N: sample size, default 500
        :param cause: boolean that indicates if there is a causal relation x1->x3
        :param noise_type: noise type ('Gaussian' or 'Uniform')
        :param num_confounders: number of confounders
        :param num_latent: number of hidden confounders
        :param nd: noise depedence factor
        :param ratio2noise: boolean that indicated that we will have a cause with multiple strengths (only possible when cause==True)
        :param additive_noise: boolean indicating if there must be additive noise
    """

    # initialize
    x1, noiseX1 = noise('Gaussian', N, track=True)
    x1 = nd * x1
    x2 = None
    x3, noiseX3 = noise('Gaussian', N, track=True)
    x3 = nd * x3
    x3s = None

    # add confounding effect
    confounders = []
    for i in range(num_confounders+num_latent):
        confounders.append(noise('Gaussian', N))
        x1 += function(confounders[-1], linear=True)
        x3 += function(confounders[-1], linear=True)

    # add cause x1->x3 on strength
    if cause and ratio2noise:
        # none = almost no cause,..., extreme= almost no noise
        strengths = {'none':0.01,'weak':.25, 'neutral':.5, 'strong':.75, 'extreme':.99}
        x3s = {}
        for strength in strengths.keys():
            x3s[strength] = (1-strengths[strength])*x3 + strengths[strength]*function(x1, linear=True)
    elif cause:
        x3 += function(x1, linear=True) + x3

    # normalize x1
    x1 = normalizeData(x1)

    # normalize observed confounders
    if num_confounders>0:
        x2 = normalizeData(confounders[0])
        for i in range(1,num_confounders):
            x2 = np.append(x2,normalizeData(confounders[i]),1)

    # normalize x3 and return
    if x3s is not None:
        for strength in strengths:
            x3s[strength] = normalizeData(x3s[strength])
        return x1, x2, x3s
    elif x3 is not None:
        x3 = normalizeData(x3)
        return x1, x2, x3


def generateGPData(graph, N, linear=False, NOISE=None, additive_noise=True):
    """
    Generate a data set with a Gaussian Process given a causal graph.
        :param graph: CausalGraph object
        :param N: sample size
        :param linear: boolean indicating 
        :param NOISE: string indicating what type of noise ('Gaussian' or 'Uniform')
        :param additive_noise: boolean indicating to generate additive noise
    """

    # Tarjans Algorithm returns the strongly connected components in topological order
    SCCs = TarjansAlgorithm(graph)

    # check for acyclicity
    for SCC in SCCs:
        if len(SCC)>1:
            raise ValueError('Input is cyclic graph.')

    # define data hidden confounders for each pair of spouses
    confounders = generateHiddenConfounders(graph, N, NOISE)

    data_dict = {}

    # generate data according to topological order
    for SCC in SCCs:

        # select variabele (strongly connected component consists of one vertex)
        variable = SCC[0]

        # select all causes
        parents = graph.parents(variable)
        parent_data = tuple(data_dict[parent] for parent in parents)
        confounder_data = tuple(confounders[variable][spouse] for spouse in confounders[variable])
        cause_data = parent_data + confounder_data

        # generate data
        data_dict[variable] =  GaussianProcessEffect(N, cause_data, NOISE, additive_noise)

    nan=False
    for variable in graph.all_variables:

        # normalize data
        data_dict[variable] = normalizeData(data_dict[variable])

        # check for nan values
        if math.isnan(data_dict[variable][0,0]) or math.isinf(data_dict[variable][0,0]):
            nan=True

    return nan, data_dict


def generateData(graph, N, linear=False, NOISE=None, avoid_lin_gaus=True):
    """
    Generate a data set given a causal graph. All relationships are additive.
        :param graph: CausalGraph
        :param N: sample size
        :param linear: boolean indicating if there are only linear relationships
        :param NOISE: type of noise ('Gaussian' or 'Uniform')
        :param avoid_lin_gaus: boolean indicating if we want to avoid linear Gaussian relationships
    """

    # check if assumptions are in line with requests.
    if avoid_lin_gaus and (linear and NOISE=='Gaussian'):
        raise ValueError('Request for linear Gaussian data while we have to avoid this type of data. You might want to reset avoid_lin_gaus.')

    # Tarjans Algorithm returns the strongly connected components in topological order
    SCCs = TarjansAlgorithm(graph)

    # check for acyclicity
    for SCC in SCCs:
        if len(SCC)>1:
            raise ValueError('Input is cyclic graph.')

    # define data hidden confounders for each pair of spouses
    confounders = generateHiddenConfounders(graph, N, NOISE)

    data_dict = {}

    # generate data according to topological order
    for SCC in SCCs:

        # select variabele (strongly connected component consists of one vertex)
        variable = SCC[0]

        # tracking the noise type ensures to avoid linear Gaussian data
        if avoid_lin_gaus:
            data_dict[variable], noise_type = noise(NOISE, N, track=True)
        else:
            data_dict[variable] = noise(NOISE, N)
            noise_type = NOISE

        # select parents of variable
        parents = graph.parents(variable)

        # add the functional relationship over the parent
        for parent in parents:
            data_dict[variable] += coefficient() * function(data_dict[parent], noise_type=noise_type)

        # add the functional relationship over the spouse (aka hidden confounder)
        for spouse in confounders[variable]:
            data_dict[variable] += coefficient('strong') * function(confounders[variable][spouse], noise_type=noise_type)
        
    nan=False
    for variable in graph.all_variables:

        # normalize data
        data_dict[variable] = normalizeData(data_dict[variable])

        # check for nan values
        if math.isnan(data_dict[variable][0,0]) or math.isinf(data_dict[variable][0,0]):
            nan=True

    return nan, data_dict


def generateHiddenConfounders(graph, N, NOISE):
    """
    In this method, we generate data of all hidden confounders in the graph:
    for each bidirected edge we define a hidden confounder.
        :param graph: CausalGraph object
        :param N: sample size
        :param NOISE: type of noise ('Gaussian' or 'Uniform')
    """

    confounders = {}

    for variable in graph.all_variables:

        # define keys
        if variable not in confounders.keys():
            confounders[variable] = {}

        for neighbour in graph.adjacencies(variable):

            # check if neighbour is a spouse
            if graph.incomingArrowType(variable, neighbour) == graph.head and graph.incomingArrowType(neighbour, variable) == graph.head:

                # check if the hidden confounder is not defined yet
                if neighbour not in confounders[variable].keys():

                    # define the hidden confounder data
                    confounders[variable][neighbour] = noise(NOISE, N)

                    # define it for the unordered pair {variable, neighbour}
                    if neighbour not in confounders.keys():
                        confounders[neighbour] = {}

                    confounders[neighbour][variable] = confounders[variable][neighbour]

    return confounders


def generateMultipleDataSets(N, E, V, name, latent_rate=.2, linear=False, dir="../generated_data/", GP=True):
    """
    In this method, we generate E data sets based on a random causal structure which is a MAG.
    We save the data, adjacency matrix and separating sets.
        :param N: sample size
        :param E: number of data sets
        :param V: number of vertices
        :param name: name of the experiment
        :param latent_rate: probability of edge being bidirected
        :param linear: boolean indicating if all relationships should be linear
        :param dir: directory
        :param GP: boolean to indicate to generate data via Gaussian Process
    """

    i = 1
    wrong_graphs = 0

    while i <= E:

        nan=True
        while nan:

            # generate MAG
            graph = mk_mag(V, p_bi=latent_rate)
            
            # generate data
            if GP:
                nan, dic = generateGPData(graph, N, linear=linear)
            else:
                nan, dic = generateData(graph, N, linear=linear)

        # save data
        dic_temp = {}
        for key in dic.keys():
            dic_temp[key] = dic[key][:,0]
        path = dir + name + "/data" + str(i) + ".csv"

        # create directory if it doesnt exist
        directory = dir + name
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save data
        pd.DataFrame.from_dict(dic_temp).to_csv(path, index=False)

        # save adjacency matrix
        path_adj_matrix = dir + name + "/adj_mat" + str(i) + ".csv"
        pd.DataFrame(graph.adjacencyMatrix(), columns=graph.all_variables, index=graph.all_variables).to_csv(path_adj_matrix)

        # find sepsets
        sepsets = findSepSets(graph)

        # save sepsets
        file = open(dir + name + "/sepsets" + str(i) + ".json", "w")
        json.dump(sepsets, file)
        file.close()

        i += 1


def generateSingleDataSet(N=400, V=10, latent_rate=.2, linear=False, GP=True):
    """
    Returns a random generated MAG and a single data set generated by following 
    the causal structure implied by this MAG.
        :param N: sample size
        :param V: number of vertices
        :param latent_rate: probability of edge being bidirected
        :param linear: boolean indicating if data should have linear relationships
        :param GP: boolean indicating if data should be generated via a Gaussian Process
    """

    nan=True
    while nan:

        # generate MAG
        graph = mk_mag(V, p_bi=latent_rate)

        # generate data
        if GP:
            nan, dic = generateGPData(graph, N, linear=linear)
        else:
            nan, dic = generateData(graph, N, linear=linear)

    return dic, graph


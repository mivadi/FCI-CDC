import statistics
from causalGraph import CausalGraph
from kernelFunctions import *
from Fishers_z_transform import condIndFisherZ
from KCIT import independenceTest
import pandas as pd


class IndependenceTest(object):

    def __init__(self, alpha=0.05, indTest='Fisher', oracle=None):

        self.alpha = alpha
        self.oracle = oracle

        if not self.oracle is None:
            self.execute = self.oracle_test
        if indTest == 'Fisher':
            self.execute = self.condIndFisherZ_test
        elif indTest == 'kernel':
            self.execute = self.KCIT_test
        

    def oracle_test(self, graph, X, Y, condition=[]):
        """
        Decide on independence making use of oracle information.
            :param graph: CausalGraph object
            :param X, Y: variables
            :param condition: list of variables (default is empty list [])
        """
        independent = False
        if X in oracle.keys() and Y in oracle[X].keys():
            for sep_set in oracle[X][Y]:
                if set(sep_set).issubset(set(condition)):
                    independent =  True
                    break
        elif Y in oracle.keys() and X in oracle[Y].keys():
            for sep_set in oracle[Y][X]:
                if set(sep_set).issubset(set(condition)):
                    independent =  True
                    break


    def condIndFisherZ_test(self, graph, X, Y, condition=[]):
        """
        Returns boolean if X and Y are independent given condition based on Fisher-Z test.
            :param graph: CausalGraph object
            :param X: variable
            :param Y: variable
            :param condition: list of variables (default is empty list [] )
        """

        # compute p-value
        p_val, _ = condIndFisherZ(X, Y, condition, graph.corr, graph.data_size)

        # if the p-value is smaller than a value alpha, we will reject the null-hypothesis
        # and except the alternative hypothesis, i.e. the variables are dependent
        if p_val > self.alpha:
            independent = True

        # otherwise we cannot reject the null-hypothesis
        else:
            independent = False

        return independent


    def KCIT_test(self, graph, X, Y, condition=[], nr_samples=50000):
        """
        Returns boolean if X and Y are independent given condition based on:
        Kernel-based Conditional Independence Test and Application in Causal Discovery - Zhang et al. (2011)
            :param graph: CausalGraph object
            :param X: variable
            :param Y: variable
            :param condition: list of variables (default is empty list [] )
            :param nr_samples: number of samples to compute the statistics
        """
        return independenceTest(graph, X, Y, condition=condition, alpha=self.alpha, nr_samples=nr_samples)

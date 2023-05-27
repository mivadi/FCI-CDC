from causalGraph import CausalGraph
from searchFunctions import *
from boolFunctions import isSeparated
from FCIFunctions import zhangsInferenceRules, completeDirectedEdges
from kernelFunctions import kernel, centralize, symmetric
from KCIT import KUIT
from sklearn.neural_network import MLPRegressor
import copy
import numpy as np
import statistics
from itertools import combinations


def CDC(X, Y, Z=None):
    """
    Compute Conditional CDC score for direction Y->X with deconfounding set Z
    using kernels of the random variables P(X|YZ) and P(Y|Z).
        :param X: variable
        :param Y: variable
        :param Z: set of deconfounding variables, optional
    """

    N = X.shape[0]
    nr_samples = 500000

    if Z is None:
        XZ = np.copy(X)
        YZ = np.copy(Y)
    else:
        XZ = np.append(X, Z/2, 1)
        YZ = np.append(Y, Z/2, 1)

    # kernel of residual Y regressed on X and Z
    regYXZ = MLPRegressor(learning_rate='adaptive',learning_rate_init=0.01, max_iter=1000).fit(XZ,Y[:,0])
    residualsYXZ = np.abs(Y - regYXZ.predict(XZ)[:,None])
    width_YX = statistics.median(np.abs(residualsYXZ - X))[0]
    width_YX = 2 / (width_YX**2)
    kernel_YXZ = symmetric(centralize(kernel(residualsYXZ, N, width_YX)))

    # kernel of residual X regressed on Y and Z
    regXYZ = MLPRegressor(learning_rate='adaptive',learning_rate_init=0.01, max_iter=1000).fit(YZ,X[:,0])
    residualsXYZ = np.abs(X - regXYZ.predict(YZ)[:,None])
    width_XY = statistics.median(np.abs(residualsXYZ - Y))[0]
    width_XY = 2 / (width_XY**2)
    kernel_XYZ = symmetric(centralize(kernel(residualsXYZ, N, width_XY)))

    # compute kernels for X and Y
    kernel_Y = symmetric(centralize(kernel(Y, N, width_YX)))
    kernel_X = symmetric(centralize(kernel(X, N, width_XY)))

    # compute unconditional independence between X and Y|X, Y and X|Y
    X_YX = KUIT(kernel_X, kernel_YXZ, nr_samples)
    Y_XY = KUIT(kernel_Y, kernel_XYZ, nr_samples)

    return X_YX, Y_XY


def selectPossibleDeconfoundingSets(PAG, X, Y):
    """
    Returns the definite and possible deconfounding set (which are disjoint sets).
        :param PAG: causalGraph object
        :param X/Y: vertex
    """

    # D1: select possible parents
    possible_parents = possibleParents(PAG, Y)
    if X in possible_parents:
        possible_parents.remove(X)

    # compute MAG as in Theorem 2 [Zhang (2008)]
    MAG = PAG2MAG(PAG)

    # remove edges adjacent to Y
    adjacencies = MAG.adjacencies(Y)
    for neighbour in adjacencies:
        MAG.deleteEdge(Y, neighbour)

    # find ancestors
    ancestors = findAncestors(MAG)

    # D2: select possible parents which are relevant for m-separation
    possible_parents_copy = possible_parents.copy()
    for parent in possible_parents:
        possible_parents_copy.remove(parent)
        if not isSeparated(MAG, parent, X, possible_parents_copy, ancestors):
            possible_parents_copy.append(parent)

    # D3: select the set satisfying Lemma 3 (i)
    def_deconf = [] # P
    pos_deconf = [] # Q \ P
    for parent in possible_parents_copy:
        if lemma_3_i(PAG, X, Y, parent):
            def_deconf.append(parent)
        else:
            pos_deconf.append(parent)

    return def_deconf, pos_deconf


def possibleParents(graph, Y):
    """
    Returns all possible parents (including definite parents).
        :param graph: causalGraph object
        :param Y: vertex
    """
    possible_parents = []
    for Z in graph.adjacencies(Y):
        if graph.incomingArrowType(Y,Z)!=graph.tail and graph.incomingArrowType(Z,Y)!=graph.head:
            # Z o-o Y, Z o-> Y, Z -> Y
            possible_parents.append(Z)
    return possible_parents


def PAG2MAG(PAG):
    """
    Theorem 2 Zhang&co (2008)
        :param PAG: causalGraph object
    """
    MAG = copy.deepcopy(PAG)
    unknown_edges = {}
    unknown_adj = {}

    for A in MAG.all_variables:
        for B in MAG.adjacencies(A):

            if MAG.incomingArrowType(A,B)==MAG.unknown and MAG.incomingArrowType(B,A)==MAG.head:
                # if A o-> B, then A -> B
                MAG.updateCause(A, B)

            elif MAG.incomingArrowType(A,B)==MAG.unknown and MAG.incomingArrowType(B,A)==MAG.tail:
                # if A o- B, then A <- B
                MAG.updateCause(B, A)

            elif MAG.incomingArrowType(A,B)==MAG.unknown and MAG.incomingArrowType(B,A)==MAG.unknown:

                # track number of unknown edges
                if A not in unknown_edges.keys():
                    unknown_edges[A] = 0
                unknown_edges[A] +=1

                # track vertex adjacent to unknown edge
                if A not in unknown_adj.keys():
                    unknown_adj[A] = []
                unknown_adj[A].append(B)

    new_parent_edges = {}
    while len(unknown_edges)>0:

        if len(new_parent_edges)==0:

            # select (as initial) vertex adjacent to the most o-o edges
            A = max(unknown_edges, key=unknown_edges.get)

        else:

            # select vertex with most new oriented parents
            A = max(new_parent_edges, key=new_parent_edges.get)

            # delete key
            del new_parent_edges[A]

        for B in unknown_adj[A]:

            # update cause
            MAG.updateCause(A, B)

            # update number of new parents into B
            if B not in new_parent_edges.keys():
                new_parent_edges[B] = 0
            new_parent_edges[B]+=1

            # delete vertex from unknown edge adjacent to B
            unknown_adj[B].remove(A)

        # delete vertex from unknown edge
        del unknown_edges[A]

    return MAG


def lemma_3_i(graph, X, Y, parent):
    """
    Check if Lemma 4.4.i holds in Diepen&co (2023).
        :param graph: causalGraph object
        :param X/Y/parent: vertex
    """
    lemma_i = False
    if parent in graph.adjacencies(X) and parent in graph.adjacencies(Y):
        if graph.isDirectedEdge(parent, X) and graph.isDirectedEdge(parent, Y):
            lemma_i = True
        elif graph.isDirectedEdge(parent, Y) and graph.incomingArrowType(parent,X)==graph.unknown and graph.incomingArrowType(X,parent)==graph.head:
            lemma_i = True
        elif graph.isDirectedEdge(parent, X) and graph.incomingArrowType(parent,Y)==graph.unknown and graph.incomingArrowType(Y,parent)==graph.head:
            lemma_i = True
    return lemma_i


def CDCorientation(graph, X, Y, threshold=.05, conservative=False, fixed_direction=False, track_deconf=False, visited_deconf=[], oracle=None):
    """
    Returns the orientation A->B given Z and deconfs.
        :param graph: causalGraph
        :param X/Y: vertex
        :param threshold: float between 0 and 1
        :param conservative: boolean whether we do a conservative check
        :param fixed_direction: boolean whether the direction is only one way possible
        :param track_deconf: boolean whether we will track the deconfounding sets
        :param visited_deconf: list of already visited deconfounding sets
        :param oracle: oracle information (causalGraph or None(=default))
    """
    if oracle is None:
        A, B, Z, deconfs = CDCorientationPred(graph, X, Y, threshold=threshold, conservative=conservative, fixed_direction=fixed_direction, track_deconf=track_deconf, visited_deconf=visited_deconf)
    else:
        A, B, Z, deconfs = CDCorientationOracle(X, Y, oracle)
    return A, B, Z, deconfs


def CDCorientationOracle(X, Y, oracle):
    """
    Returns the orientation A->B given Z given the oracle causalGraph.
        :param X/Y: vertex
        :param oracle: oracle information (causalGraph or None(=default))
    """

    ancestors = findAncestors(oracle)
    orientation = False

    if X in oracle.adjacencies(Y):

        # check for X-> Y in oracle MAG
        if oracle.isDirectedEdge(X, Y):
            A, B = X, Y
            orientation = True
        # check for Y->X in oracle MAG
        elif oracle.isDirectedEdge(Y, X):
            A, B = Y, X
            orientation = True

    # select deconfounding set in oracle MAG if there is a directed egde
    if orientation:

        # select parents of B that are ancestors of A
        Z = []
        for adj in oracle.adjacencies(B):
            if oracle.isDirectedEdge(adj, B) and adj in ancestors[A]: ## adj -> B and ancestor of A
                Z.append(adj)
    else:
        A, B, Z = None, None, None

    return A, B, Z, []


def CDCorientationPred(graph, X, Y, threshold=.05, conservative=False, fixed_direction=False, track_deconf=False, visited_deconf=[]):
    """
    Returns the orientation A->B given Z and deconfs.
        :param graph: causalGraph
        :param X/Y: vertex
        :param threshold: float between 0 and 1
        :param conservative: boolean whether we do a conservative check
        :param fixed_direction: boolean whether the direction is only one way possible (X->Y)
        :param track_deconf: boolean whether we will track the deconfounding sets
        :param visited_deconf: list of already visited deconfounding sets
    """

    # select definite and possible deconfounding sets.
    def_deconf, pos_deconf = selectPossibleDeconfoundingSets(graph, X, Y)

    # initilize
    orientation = False
    A, B, Z = None, None, None
    p = 0
    deconfs = []
    if conservative:
        track_orientations = {X:{Y:0},Y:{X:0}}
        track_deconf = {X:{Y:[]},Y:{X:[]}}

    # test for all orientations as long as we did not find an orientation
    # and as long as we can increase the deconfounding set
    while not orientation and p <= len(pos_deconf):

        pos_deconf_subsets = combinations(pos_deconf, p)
        p+=1

        for pos_deconf_subset in pos_deconf_subsets:

            # D4: select possible deconfounding set
            Z = [*pos_deconf_subset] + def_deconf

            # track the visited possible deconfounding sets
            if track_deconf:
                deconfs.append(set(Z))

            # if we already visited Z we go to the next Z
            if set(Z) in visited_deconf: continue

            # test CDC given selected possible deconfounding set
            X_YX, Y_XY = CDC(graph.getData([X]), graph.getData([Y]), graph.getData(Z))

            # check for X -> Y orientation
            if X_YX > threshold and Y_XY < threshold:
                if conservative:
                    # track deconfounding variables
                    if track_orientations[X][Y] == 0:
                        track_deconf[X][Y] = list(Z)
                    else:
                        track_deconf[X][Y] = list(set(track_deconf[X][Y] + Z))
                    # track the orientations for the conservative-CDC
                    track_orientations[X][Y] += 1
                else:
                    orientation = True
                    A, B = X, Y
                    # orientation is found; break for-loop
                    break

            # if the other direction seems to be true
            elif X_YX < threshold and Y_XY > threshold:
                if not fixed_direction:
                    if conservative:
                        # track deconfounding variables
                        if track_orientations[Y][X] == 0:
                            track_deconf[Y][X] = list(Z)
                        else:
                            track_deconf[Y][X] = list(set(track_deconf[Y][X] + Z))
                        # track the orientations for the conservative-CDC
                        track_orientations[Y][X] += 1
                    else:
                        # we only can orient this side if the direction is not fixed
                        orientation = True
                        A, B = Y, X
                        break

    if conservative:
        # very strict check if only one orientation direction is possible
        if track_orientations[X][Y] > 0 and track_orientations[Y][X]  == 0:
            A, B, Z = X, Y, track_deconf[X][Y] # no parient orientation?
        elif track_orientations[Y][X] > 0 and track_orientations[X][Y]  == 0:
            A, B, Z = Y, X, track_deconf[Y][X] # no parent orientation?

    return A, B, Z, deconfs


def parentOrientationRule(graph, X, parents):
    """
    Orients all parents according to parent orientation rule in FCI-CDC.
        :param graph: causalGraph
        :param X: vertex
        :param parents: list of vertices
    """
    for parent in parents:
        if parent in graph.adjacencies(X):
            graph.updateCause(parent, X, rule='parent')


def CDCrules(graph, separating_sets, oracle=None, beta=.05, conservative=False):

    """
        :param graph: CausalGraph
        :param separated_sets: dictionary containing all separating sets
        :param threshold: threshold for the independence tests for CDC
        :param conservative: boolean whether we will run conservative-CDC
        :param oracle: CausalGraph
    """

    # track orientations made by CDC and additional rules
    graph.getCDCorientationMatrices()

    visited = []
    for X in graph.all_variables:
        visited.append(X)
        for Y in graph.adjacencies(X):
            if Y in visited: continue

            A = None

            # X o-> Y
            if graph.incomingArrowType(X,Y)==graph.unknown and graph.incomingArrowType(Y,X)==graph.head:
                # search for X -> Y orientation
                A, B, Z, _ = CDCorientation(graph, X, Y, threshold=beta, conservative=conservative, fixed_direction=True, oracle=oracle)
                # make sure that we did not orient in the opposite direction
                if A == Y: A=None
            # X <-o Y
            elif graph.incomingArrowType(X,Y)==graph.head and graph.incomingArrowType(Y,X)==graph.unknown:
                # search for X <- Y orientation
                A, B, Z, _ = CDCorientation(graph, Y, X, threshold=beta, conservative=conservative, fixed_direction=True, oracle=oracle)
                # make sure that we did not orient in the opposite direction
                if A == X: A=None
            # X o-o Y
            elif graph.incomingArrowType(X,Y)==graph.unknown and graph.incomingArrowType(Y,X)==graph.unknown:
                # search for X -> Y orientation
                A, B, Z, deconfs = CDCorientation(graph, X, Y, threshold=beta, conservative=conservative, fixed_direction=False, track_deconf=True, oracle=oracle)
                if A is None:
                    # search for X <- Y orientation
                    A, B, Z, _ = CDCorientation(graph, Y, X, threshold=beta, conservative=conservative, fixed_direction=False, visited_deconf=deconfs, oracle=oracle)

            # if an orientation has been found
            if not A is None:

                # update edge: A -> B
                graph.updateCause(A, B, rule='CDC')

                # orient parents
                parentOrientationRule(graph, B, Z)

                # Zhangs version of Meeks orientation rules: R1-R4'
                zhangsInferenceRules(graph, separating_sets, CDC=True)

                # complete directed edges: R8-R10
                completeDirectedEdges(graph, CDC=True)



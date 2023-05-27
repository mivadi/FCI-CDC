from causalGraph import CausalGraph
from itertools import combinations
from boolFunctions import *
import copy


def findAdjXnotY(adj, Y):
    """
    Returns all adjacencies excluding Y.
        :param adj: list of adjacencies of X
        :param Y: variable
    """
    # all adjacencies
    adjX = adj[:]

    # remove Y from adjacencies
    adjX.remove(Y)
    return adjX


def findSubpaths(path):
    """
    Returns all subpaths of path starting from length 3.
        :param path: list of variables that form a path in a graph
    """
    subpaths = []
    if len(path) > 2:
        # add all subpaths of all possible lenghts starting with length 3
        for i in range(len(path)-2):
            subpaths.append((path[i],path[i+1],path[i+2]))
    return subpaths


def findDirectedPaths(graph, variable1, variable2):
    """
    Returns all directed paths from variable 1 to variable 2 in graph.
        :param graph: causalGraph
        :param variable1/variable2: vertex in graph
    """

    paths = [[variable1]]
    final_paths = []
    found_all_paths = False

    # search through all possible paths as long as it contains only directed edges
    while not found_all_paths:
        found_all_paths = True
        new_paths = []
        for path in paths:
            adjacency_variables = graph.adjacencies(path[-1])
            for next_variable in adjacency_variables:
                # check if there is a directed edge
                if graph.isDirectedEdge(path[-1], next_variable):
                    new_path = path.copy()
                    new_path.append(next_variable)
                    if next_variable not in path:
                        new_paths.append(new_path)
                        found_all_paths = False
                    elif next_variable == variable2:
                        final_paths.append(new_path)

        paths = new_paths
    return final_paths


def findPaths(graph, variable1, variable2, track_colliders=False):
    """
    Returns all paths between variable 1 and variable 2 in graph.
        :param graph: causalGraph
        :param variable1/variable2: vertex in graph
        :param track_colliders: boolean that is true if we need to track the
                                colliders for efficiency
    """

    paths = [[variable1]]
    if track_colliders:
        paths_colliders = [[False]]

    final_paths = []

    found_all_paths = False
    while not found_all_paths:
        found_all_paths = True
        new_paths = []
        new_paths_colliders = []

        for i, path in enumerate(paths):

            adjacency_variables = graph.adjacencies(path[-1])

            for next_variable in adjacency_variables:

                if next_variable not in path:

                    new_path = path.copy()
                    new_path.append(next_variable)

                    if track_colliders:
                        new_path_colliders = paths_colliders[i].copy()

                        # if the last variable in the path is a possible collider
                        if new_path_colliders[-1]:
                            # but it has no incoming arrow head on the other side
                            if graph.incomingArrowType(path[-1], next_variable) == graph.tail:
                                # then it is a non-collider
                                new_path_colliders[-1] = False

                    if next_variable == variable2:
                        if track_colliders:
                            # last element of the path is not a collider
                            new_path_colliders.append(False)
                            final_paths.append((new_path, new_path_colliders))
                        else:
                            final_paths.append(new_path)
                    else:
                        if track_colliders:
                            if graph.incomingArrowType(next_variable, path[-1]) == graph.head:
                                # possible collider
                                new_path_colliders.append(True)
                            else:
                                # non-collider
                                new_path_colliders.append(False)
                            new_paths_colliders.append(new_path_colliders)
                        new_paths.append(new_path)
                        found_all_paths = False

        paths = new_paths
        if track_colliders:
            paths_colliders = new_paths_colliders

    return final_paths


def findStronglyConnectedComponents(graph, acyclic):
    """
    Returns the strongly connected components in graph.
    ONLY USE THIS IF ACYCLIC: otherwise use Tarjans algorithms (TODO: connect the two functions)
        :param graph: causalGraph
        :param acyclic: boolean that is true when graph is acyclic
    """

    SCCs = {}
    visited_variables = []

    for variable in graph.all_variables:

        # in case we consider a acyclic
        if acyclic:
            SCCs[variable] = [variable]
            continue

        # continue if variable is already visited
        if variable in SCCs.keys(): continue

        # find current strongly connected component
        scc = [variable]
        directed_paths = findDirectedPaths(graph, variable, variable)

        # print(directed_paths)
        for path in directed_paths:
            scc = scc + path
        scc = list(set(scc))

        # save strongly connected component in dictionary for all each variables
        for var in scc:
            SCCs[var] = scc

    return SCCs


def findSepSets(graph, acyclic=True, SCCs=None):
    """
    Returns separating sets in graph.
        :param graph: causalGraph
        :param acyclic: boolean that is true when graph is acyclic
        :param SCCs: are the strongly connected components (default is None, in
                    that case we will search for the strongly connected components)
    """

    sep_sets = {}

    # find strongly connected components
    if SCCs is None:
        SCCs = findStronglyConnectedComponents(graph, acyclic)

    ancestors = findAncestors(graph)
    variables = graph.all_variables.copy()
    # loop over all pairs of variables
    for i in range(len(variables)-1):
        for j in range(i+1, len(variables)):
            # not separated if the variables are adjacent
            if variables[j] in graph.adjacencies(variables[i]): continue
            # not separated if the variables are in the same strongly connected component
            if variables[j] in SCCs[variables[i]]: continue
            # search for separating set between current variables
            sep_set = findSepSet(graph, variables[i], variables[j], SCCs, ancestors, acyclic)
            # save separating set
            if sep_set is not None:
                if variables[i] not in sep_sets.keys():
                    sep_sets[variables[i]] = {}
                if variables[j] not in sep_sets[variables[i]].keys():
                    sep_sets[variables[i]][variables[j]] = []
                sep_sets[variables[i]][variables[j]].append(sep_set)
    return sep_sets


def findSepSet(graph, A, B, SCCs, ancestors, acyclic):
    """
    Returns separating sets in graph.
        :param graph: causalGraph
        :param A/B: vertices
        :param SCCs: are the strongly connected components (default is None, in
                    that case we will search for the strongly connected components)
        :ancestors: dictionary with all ancestors for all vertices
        :param acyclic: boolean that is true when graph is a directed acyclic graph
    """

    other_variables = graph.all_variables.copy()
    other_variables.remove(A)
    other_variables.remove(B)

    # search through possible separation sets
    for n in range(len(other_variables)+1):
        possible_sep_sets = combinations(other_variables, n)
        for possible_sep_set in possible_sep_sets:
            if isSeparated(graph, A, B, possible_sep_set, ancestors, acyclic=acyclic, SCCs=SCCs):
                return list(possible_sep_set)
    return None


def findDeconfounder(graph, DAG=False):
    """
    Returns deconfounders between all adjacent variables in graph.
        :param graph: CausalGraph
        :param DAG: boolean that is true when graph is a directed acyclic graph
    """

    # find strongly connected components
    SCCs = findStronglyConnectedComponents(graph, DAG)
    deconfounders = {}

    # loop over all adjacent vertices
    for A in graph.variables:
        for B in graph.adjacencies(A):
            # check if there can be an deconfounding set
            if graph.isDirectedEdge(A, B) and not B in SCCs[A]: # A->B
                # delete joint edge
                temp_graph = copy.deepcopy(graph)
                temp_graph.deleteEdge(A,B)
                ancestors = findAncestors(temp_graph)
                if A not in deconfounders.keys():
                    deconfounders[A] = {}
                if B not in deconfounders[A].keys():
                    # search for separating set
                    deconfounders[A][B] = findSepSet(temp_graph, A, B, SCCs, ancestors, DAG)

    return deconfounders


def findAncestors(graph):
    """
    Returns ancestors of all vertices in graph.
        :param graph: CausalGraph
    """

    ancestors = {}

    # loop over all variables
    for variable in graph.all_variables:
        found_all_ancestors = False
        # include current variable in ancestor set
        ancestors[variable] = [variable]
        incoming_variables = [variable]
        # search for more ancestors
        while not found_all_ancestors:
            found_all_ancestors = True
            new_incoming_variables = []
            for inc_variable in incoming_variables:
                # loop over all adjacencies
                for adj_variable in graph.adjacencies(inc_variable):
                    # check if they have an incoming edge aka are ancestors
                    if graph.isDirectedEdge(adj_variable, inc_variable):
                        if adj_variable not in ancestors[variable]:
                            # add to ancestors
                            found_all_ancestors = False
                            ancestors[variable].append(adj_variable)
                            new_incoming_variables.append(adj_variable)
            incoming_variables = new_incoming_variables

    return ancestors


def findStartDiscriminatingPath(graph, A, B, C):
    """
    Returns vertex D corresponding to a discriminating path <D,...,A,B,C> if it
    exists and None otherwise.
        :param graph: CausalGraph
        :param A/B/C: vertices such that A, B and C are a possible end of a
                    discriminating path
    """

    parents = []
    possible_starts = graph.all_variables.copy()
    possible_starts.remove(C)
    for variable in graph.adjacencies(C):
        # search for possible start for the discriminating path
        possible_starts.remove(variable)
        if graph.isDirectedEdge(variable, C):
            # select the parents of C
            parents.append(variable)

    # loop over possible starts for the discriminating path
    for D in possible_starts:
        visited_variables = [A,B,C,D]
        paths = [[D]]
        # search over paths
        while paths != []:
            new_paths = []
            for path in paths:
                for variable in parents:
                    # check if variable is possible next vertex on the path
                    if variable in graph.adjacencies(path[-1]) and variable not in path:
                        # check if all vertices between D and A are colliders
                        if graph.incomingArrowType(variable, path[-1]) == graph.head:
                            if path[-1] == D or graph.incomingArrowType(path[-1], variable) == graph.head:
                                # if arrived at A, we found a discriminating path
                                if variable == A:
                                    return D
                                else:
                                    new_path = path.copy()
                                    new_path.append(variable)
                                    new_paths.append(new_path)
            paths = new_paths

    return None


def findUncoveredCirclePath(graph, A, B, C, D):
    """
    Returns uncovered circle paths <A,B,...,C,D> in graph.
        :param graph: CausalGraph
        :param A/B/C/D: vertices in graph s.t. the pair (A,B) and (C,D) both
                        share an o-o edge.
    """

    paths = [[A, C]]
    final_paths = []

    # search for all possible paths
    while paths != []:
        new_paths = []
        for path in paths:
            for variable in graph.adjacencies(path[-1]):
                if variable not in [A,B] and variable not in path:
                    # check if the triple is unshielded
                    if path[-2] not in graph.adjacencies(variable):
                        # check if there is an o-o edge connecting the variables
                        if graph.incomingArrowType(variable, path[-1]) == graph.unknown and graph.incomingArrowType(path[-1], variable) == graph.unknown:
                            new_path = path.copy()
                            new_path.append(variable)
                            if variable == D:
                                if path[-1] not in graph.adjacencies(B):
                                    # we found an uncovered circle path
                                    new_path.append(B)
                                    final_paths.append(new_path)
                            else:
                                new_paths.append(new_path)
        paths = new_paths

    return final_paths


def findUncoveredPotentiallyDirectedPath(graph, A, B):
    """
    Return uncovered potentially directed paths from A to B in graph.
        :param graph: CausalGraph
        :param A/B: vertices
    """

    paths = [[A]]
    final_paths = []

    # search for all possible uncovered potentially directed paths
    while paths != []:
        new_paths = []
        for path in paths:
            for variable in graph.adjacencies(path[-1]):
                if variable not in path:
                    # check for uncovered / unshielded:
                    if len(path) == 1 or path[-2] not in graph.adjacencies(variable):
                        # check for potentially directed edge
                        if graph.incomingArrowType(variable, path[-1]) != graph.tail and graph.incomingArrowType(path[-1], variable) != graph.head:
                            new_path = path.copy()
                            new_path.append(variable)
                            if variable == B:
                                final_paths.append(new_path)
                            else:
                                new_paths.append(new_path)
        paths = new_paths

    return final_paths


def findMinimalDeconfounders(graph, A, B):
    """
    Return definite and possible deconfounding sets for CD-NOD (?).
        :param graph: CausalGraph
        :param A/B: adjacent vertices
    """

    # initialize (possible) minimal deconfounding set
    MD, PMD = [], []
    adj_A = graph.adjacencies(A)
    # loop over all vertices adjacent to A
    for Z in adj_A:
        if Z is not B and Z is not graph.context:
            # if Z->A, Z is a definite deconfounder
            if graph.isDirectedEdge(Z, A):
                MD.append(Z)
            # if Z*-A or Z*-oA, Z is a possible deconfounder
            elif graph.incomingArrowType(Z, A) != graph.head:
                PMD.append(Z)

    return MD, PMD

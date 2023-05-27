from itertools import combinations, permutations, product
from collections import defaultdict
from causalGraph import CausalGraph
from searchFunctions import *
from KCIT import *
from boolFunctions import *


def deleteSeparatedEdges(graph, separating_sets, JCI='0', IT=None, oracle=None):
    """
    Learning the skeleton.
        :param graph: CausalGraph
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param JCI: string ('0', '1', '12', '123', default is '0') indicating which  
                        assumtions from 'Joint Causal Inference from Multiple Contexts'
                        by Joris Mooij et al. (2020) are assumed to be true
        :param IT: IndependenceTest object
        :param oracle: true separating_sets
    """
    # learning the skeleton

    if IT is None and oracle is None:
        raise ValueError('IndependenceTest or oracle required.')

    if oracle is None:

        depth = 0
        done = False
        while not done:
            # initilize done again
            done = True
            for X in graph.all_variables:
                # find the adjacency variables of X
                # with stable version: https://www.jmlr.org/papers/volume15/colombo14a/colombo14a.pdf
                adjacency_variables = graph.adjacencies(X)
                # find the amount of adjacency variables of X
                num_adj = len(adjacency_variables)
                # if there are less than [depth] adjacency variables excluding Y
                # then continue to the next variable X
                if num_adj-1 < depth: continue
                # otherwise we are not done
                # and we will check whether we may delete edges
                done=False
                for Y in adjacency_variables:

                    # keep the edge if assumption 3 of JCI is assumed
                    if '3' in JCI and X in graph.context and Y in graph.context: continue

                    # find the list with adjacency variable of X excluding Y
                    adjXnotY = findAdjXnotY(graph.adjacencies(X), Y)
                    # find all possible subsets of size [depth] in adjXnotY
                    subsets = combinations(adjXnotY, depth)
                    for subset in subsets:
                        # check if X and Y are independent given [subset]
                        if IT.execute(graph, X, Y, condition=subset):#, oracle=oracle):
                            # remove the edge between X and Y
                            graph.deleteEdge(X, Y)
                            # save the separating sets
                            separating_sets[(X,Y)] = subset
                            separating_sets[(Y,X)] = subset

                            break
            depth += 1
    else:
        for X in graph.all_variables:
            adjacency_variables = graph.adjacencies(X)
            for Y in adjacency_variables:
                if X in oracle.keys() and Y in oracle[X].keys():
                    subset = tuple(oracle[X][Y][0])
                    graph.deleteEdge(X, Y)
                    separating_sets[(X,Y)] = subset
                    separating_sets[(Y,X)] = subset
                elif Y in oracle.keys() and X in oracle[Y].keys():
                    subset = tuple(oracle[Y][X][0])
                    graph.deleteEdge(X, Y)
                    separating_sets[(X,Y)] = subset
                    separating_sets[(Y,X)] = subset


def jci(graph, JCI='0'):
    """
    Orientations according to 'Joint causal inference from multiple contexts' by Joris Mooij et al. (2020).
        :param graph: CausalGraph
        :param JCI: string ('0', '1', '12', '123', default is '0') indicating which  
                    assumtions from 'Joint Causal Inference from Multiple Contexts'
                    by Joris Mooij et al. (2020) are assumed to be true
    """
    if JCI != '0':
        for X in graph.context:
            for Y in graph.adjacencies(X):
                if Y in graph.context:
                    if '3' in JCI:
                        # update edge to X<->Y
                        graph.updateEdge(graph.head, X, Y)
                        graph.updateEdge(graph.head, Y, X)
                elif '1' in JCI:
                    # update edge Xo->Y
                    graph.updateEdge(graph.head, Y, X)
                    if '2' in JCI:
                        # update edge X->Y
                        graph.updateEdge(graph.tail, X, Y)


def background(graph, bkg_info=None):
    """
    Orientations according to background information.
        :param graph: CausalGraph
        :param bkg_info: background information in the form of an ancestral graph 
                    represented by a matrix where [i,j]=1 if i is an ancestor of j, 
                    [i,j]=1 if i is a non-descendants of j, and 0 means no info
    """
    if not bkg_info is None:
        for X in graph.all_variables:
            for Y in graph.adjacencies(X):
                if bkg_info.loc[X,Y] == 1 and bkg_info.loc[Y,X] == 1:
                    raise ValueError('Background information suggests cycles: redefine background or choose algorithm that works for cycles.')
                elif bkg_info.loc[X,Y] == 1:
                    # bkg_info.loc[Y,X] is 0 or -1
                    # X -> Y
                    graph.updateCause(X,Y)
                elif bkg_info.loc[Y,X] == 1:
                    # bkg_info.loc[X,Y] is 0 or -1
                    # X <- Y
                    graph.updateCause(Y,X)
                elif bkg_info.loc[X,Y] == -1 and bkg_info.loc[Y,X] == -1:
                    # X <-> Y
                    graph.updateEdge(graph.head, X, Y)
                    graph.updateEdge(graph.head, Y, X)
                elif bkg_info.loc[X,Y] == -1:
                    # bkg_info.loc[Y,X] is 0
                    # X <-* Y
                    graph.updateEdge(graph.head, X, Y)
                elif bkg_info.loc[Y,X] == -1:
                    # bkg_info.loc[X,Y] is 0
                    # X *-> Y
                    graph.updateEdge(graph.head, Y, X)


def addColliders(graph, separating_sets, PC=False, conservative=False, IT=None, uncertain_triples=None):
    """
    Rule 0 in Zhang (2008) where the colliders are added.
        :param graph: CausalGraph
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param PC: boolean (if PC, we cannot orient bidirected edges)
        :param conservative: boolean indicating whether we do an additional conservative check
        :param IT: IndependenceTest object
        :param uncertain_triples: defaultdict(list) to track the uncertain triples when conservative=True

    """
    if uncertain_triples is None and conservative:
        raise ValueError("Dictionary of uncertain triples not defined.")

    # add colliders
    for Y in graph.variables:

        # find all pairs of adjacency variables of Y
        adjacency_pairs = combinations(graph.adjacencies(Y), 2)
        for X, Z in adjacency_pairs:
            # check if X and Y are not adjacent
            if X not in graph.adjacencies(Z):
                # check if Y is not in the separating set of X and Z
                if Y not in separating_sets[(X,Z)]:

                    # check if the collider is uncertain
                    if conservative:
                        uncertain_collider = IT.execute(graph, X, Z, condition=list(separating_sets[(X,Z)])+[Y])
                        if uncertain_collider:
                            uncertain_triples[(X,Z)].append(Y)
                    else:
                        uncertain_collider = False

                    if PC: # always not conservative
                        if graph.incomingArrowType(X, Y) == graph.tail and graph.incomingArrowType(Z, Y) == graph.tail:
                            # update edge X-Y to X->Y  and edge Z-Y to Z->Y
                            graph.updateEdge(graph.head, Y, X)
                            graph.updateEdge(graph.head, Y, Z)

                    elif not uncertain_collider: # FCI
                        # update edge Xo-oY to Xo->Y  and edge Zo-oY to Zo->Y
                        graph.updateEdge(graph.head, Y, X)
                        graph.updateEdge(graph.head, Y, Z)


def addNonCollidersPC(graph):
    """
    Remaining orientation rules of PC.
        :param graph: CausalGraph
    """

    done = False
    while not done:

        # initilize done again
        done = True

        directed_edges = graph.directedEdges()
        for A, B in directed_edges:
            # find the list with adjacency variable of B excluding A
            adjBnotA = findAdjXnotY(graph.adjacencies(B), A)
            for C in adjBnotA:
                # check if C is not adjacent to A
                # and check if B does not have an incoming arrow head from C
                if C not in graph.adjacencies(A) and (C,B) not in graph.directedEdges():
                    if (B,C) not in graph.directedEdges():
                        # update the edge (B-C) to (B->C)
                        graph.updateEdge(graph.head, C, B)
                        # if an update is possible, we have to do the whole check again
                        done = False

        # NOTE: this part assumes acyclicity
        directed_paths = graph.directedPaths()
        for A, B in directed_paths:
            if B in graph.adjacencies(A):
                if (A, B) not in graph.directedEdges():
                    # update the edge (A-B) to (A->B)
                    # graph.updateCause(A, B)
                    graph.updateEdge(graph.head, B, A)
                    # if an update is possible, we have to do the whole check again
                    done = False


def deleteSeparatedEdgesStage2(graph, graph_copy, separating_sets, IT):
    """
    Edge deletion by testing for possible-D-separation sets 
    (see Causation, Prediction, and Search by Spirtes, Glymour and Scheines (1993)).
        :param graph: CausalGraph
        :param graph_copy: CausalGraph - copy of the first graph but rule R0 is executed
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param IT: IndependenceTest object
    """

    ancestors = findAncestors(graph)
    visited = defaultdict(lambda: defaultdict(lambda: False))

    for X in graph.all_variables:

        adjacencies = graph.adjacencies(X)

        for Y in adjacencies:

            # check if pair is already visited
            if visited[Y][X]: continue
            
            # update visited
            visited[Y][X] = True
            visited[X][Y] = True

            # select all paths between X and Y: only save the colliders and non-collider occuring in the paths
            colliders_in_paths = []
            pos_non_colliders_in_paths = []
            for current in graph.adjacencies(X):
                if current!=Y:
                    rest_colliders_in_paths, rest_pos_non_colliders_in_paths = searchPathsRecursive(graph_copy, current, X, Y, [X])
                    colliders_in_paths = colliders_in_paths + rest_colliders_in_paths
                    pos_non_colliders_in_paths = pos_non_colliders_in_paths + rest_pos_non_colliders_in_paths
                    # colliders_in_paths.append(rest_colliders_in_paths)
                    # pos_non_colliders_in_paths.append(rest_pos_non_colliders_in_paths)

            # we cannot select a possible d-sep set when there are no paths between X and Y (excluding the path X*-*Y)
            if len(colliders_in_paths)==0: continue

            # stack lists to matrix
            colliders = np.vstack(colliders_in_paths) # dimensions = (paths, vertices)
            non_colliders = np.vstack(pos_non_colliders_in_paths)

            # select paths that need to be blocked
            open_paths = list(np.where(np.sum(colliders,1)==0)[0])

            # initialize the list of possible separating sets per path
            pss_per_path = []

            # for each open path k select every non-collider on the path as possible separating set
            # and a temporary variable that placeholds the case that some possible non-colliders
            # are colliders, in that case we do not need to test 
            for k in open_paths:
                pss_per_path.append(list(np.where(non_colliders[k]==1)[0])+['tmp'])

            # combine the selected non-colliders to create possible separating sets that block all open paths
            combinations = list(product(*pss_per_path))

            # initialize set containing all possible d-sep-sets
            possible_d_ss = []

            # track visited pds 
            visited_pds = []

            while len(combinations)>0:

                # select and delete first possible-d-sep set
                pds = combinations.pop(0)

                # remove doubles and remove tmp variable
                pds = set(pds)-{'tmp'}

                # empty set is already tested for independence
                if len(pds)==0: continue

                pds_vars = {graph.all_variables[i] for i in pds}

                # independence is already tested when pds is a subset of the adjacencies of X xor Y
                if pds_vars <= set(graph.adjacencies(X)): continue

                if pds_vars <= set(graph.adjacencies(Y)): continue

                # check if the possible d-sep set was already tested before
                if pds in visited_pds: continue

                # otherwise add to visited pds
                visited_pds.append(pds)

                # redefine as tuple
                # pds = tuple(pds)

                # initialize Boolean for valid possible-d-sep set
                valid_pds = True

                # loop over all paths
                for k in range(colliders.shape[0]):

                    # path is blocked by non-collider
                    if 1 in non_colliders[k,tuple(pds)]: continue

                    # check if ancestors of colliders are in the possible d-sep set
                    nr_coll_k_open_by_pds = 0

                    # count number of colliders that open the path
                    cols = np.where(colliders[k,:]==1)[0]
                    for col in cols:
                        for ancestor in ancestors[graph.all_variables[col]]:
                            if ancestor in pds_vars:
                                nr_coll_k_open_by_pds += 1
                                # if collider is open: break to go to next collider
                                break

                    # check if there are other colliders blocking the path
                    if nr_coll_k_open_by_pds>0 and nr_coll_k_open_by_pds == sum(colliders[k,:]):

                        # no other collider is blocking the path, so we need to add an non-collider
                        valid_pds = False

                        # select the possible non-colliders for blocking the path
                        non_coll_k = list(set(np.where(non_colliders[k,:]==1)[0]) - pds)

                        # redefine the possible d-sep sets if there are options to block the path
                        if len(non_coll_k) > 0:

                            # add new possible colliders to the possible d-sep set
                            new_pdss = [tuple(list(pds) + [V]) for V in non_coll_k]
                            
                            # add the new possible d-sep sets to the combinations set
                            combinations = new_pdss + combinations

                            # break for-loop to restart checking for validity of the possible-d-sep sets
                            break

                # add the possible d-sep set to the list with final possible d-sep sets 
                # if all paths are possibly blocked
                if valid_pds:

                    # check if X and Y are independent given the possible d-sep set
                    if IT.execute(graph, X, Y, condition=list(pds_vars)):

                        # remove the edge between X and Y
                        graph.deleteEdge(X, Y)
                        graph_copy.deleteEdge(X, Y)

                        # save the separating sets
                        separating_sets[(X,Y)] = pds
                        separating_sets[(Y,X)] = pds

                        # path is blocked: we can break the while loop
                        break

                    else:

                        # add non-colliders to possible d-separating set if not independent

                        # select the possible non-colliders for blocking the path
                        non_coll_k = list(set(np.where(non_colliders[k,:]==1)[0]) - pds)

                        # redefine the possible d-sep sets if there are options to block the path
                        if len(non_coll_k) > 0:

                            # add new possible colliders to the possible d-sep set
                            new_pdss = [tuple(list(pds) + [V]) for V in non_coll_k]

                            # add the new possible d-sep sets to the combinations set
                            combinations = new_pdss + combinations

                            # break for-loop to restart checking for validity of the possible-d-sep sets
                            break


def searchPathsRecursive(graph, current, previous, final, visited):
    """
    Recursive function returns all possible paths where each path is an array of colliders and non-colliders. 
    The order of the vertices in the path is not important for this purpose, so we don't save the order.
        :param graph: causalGraph object
        :param current, previous: current and previous vertex on a path.
        :param final: final vertex on the path
        :param visited: all visited vertices on the path
    """

    # select new next vertices on the path
    nexts = list(set(graph.adjacencies(current)) - set(visited))

    # initialize possible paths
    colliders_in_paths = []
    pos_non_colliders_in_paths = []

    # loop over the next variables
    for next_var in nexts:

        # first we search for rest of paths
        if next_var != final:
            rest_colliders_in_paths, rest_pos_non_colliders_in_paths = searchPathsRecursive(graph, next_var, current, final, visited+[current])
        else:
            M = len(graph.all_variables)
            rest_colliders_in_paths, rest_pos_non_colliders_in_paths = [np.zeros(M)], [np.zeros(M)]
            
        # check if the current triple form a collider
        if graph.incomingArrowType(current, previous)==graph.head and graph.incomingArrowType(current, next_var)==graph.head:
            for colliders_in_path in rest_colliders_in_paths:
                colliders_in_path[graph.all_variables.index(current)] = 1
        else:
            for pos_non_colliders_in_path in rest_pos_non_colliders_in_paths:
                pos_non_colliders_in_path[graph.all_variables.index(current)] = 1
        
        colliders_in_paths = colliders_in_paths + rest_colliders_in_paths
        pos_non_colliders_in_paths = pos_non_colliders_in_paths + rest_pos_non_colliders_in_paths

    return colliders_in_paths, pos_non_colliders_in_paths


def zhangsInferenceRules(graph, separating_sets, CDC=False, uncertain_triples=None, IT=None, conservative=False):
    """
    Rules 1-4 in Zhang (2008).
        :param graph: CausalGraph
        :param separating_sets: dictionary of sets for each pair of separated vertices
        :param CDC: boolean (if CDC, R4 is slightly different and track CDC orientations)
        :param uncertain_triples: defaultdict(list) to track the uncertain triples when conservative=True
        :param IT: IndependenceTest object
        :param conservative: boolean indicating whether we do an additional conservative check
    """

    found_orientation = True

    if CDC:
        rule='FCI'
    else:
        rule=None

    while found_orientation:

        found_orientation = False

        for B in graph.variables:

            # find two adjacent variables
            adjacency_pairs = permutations(graph.adjacencies(B), 2)

            for A, C in adjacency_pairs:

                # shielded triple
                if A in graph.adjacencies(C):

                    # R2 pull through directed path
                    if graph.incomingArrowType(C, A) == graph.unknown:
                        if graph.isDirectedEdge(A, B) and graph.incomingArrowType(C, B) == graph.head:
                            graph.updateEdge(graph.head, C, A, rule)
                            # R2 update (1)
                            found_orientation = False
                        elif graph.isDirectedEdge(B, C) and graph.incomingArrowType(B, A) == graph.head:
                            graph.updateEdge(graph.head, C, A, rule)
                            # R2 update (2)
                            found_orientation = True

                    # R4 discriminating path orientation
                    if graph.incomingArrowType(B, C) == graph.unknown:

                        # A is possible collider on discriminating path
                        # A is parent of C
                        if graph.incomingArrowType(A, B) == graph.head and graph.isDirectedEdge(A, C):
                            D = findStartDiscriminatingPath(graph, A, B, C)
                            if D is not None:
                                uncertain_collider = False
                                if conservative and not CDC:
                                    if B in uncertain_triples[(C,D)] or B in uncertain_triples[(D,C)]:
                                        uncertain_collider = True
                                    elif not B in separating_sets[(C,D)]:
                                        if IT.execute(graph, C, D, condition=list(separating_sets[(C,D)])+[B]):
                                            uncertain_collider = True
                                            uncertain_triples[(C,D)] = B
                                if not (conservative and uncertain_collider):
                                    if B in separating_sets[(C,D)] or CDC:
                                        graph.updateEdge(graph.head, C, B, rule)
                                        graph.updateEdge(graph.tail, B, C, rule)
                                    else:
                                        graph.updateEdge(graph.head, C, B)
                                        graph.updateEdge(graph.head, B, C)
                                        graph.updateEdge(graph.head, A, B)
                                        graph.updateEdge(graph.head, B, A)

                # unshielded triple
                else:
                    if graph.incomingArrowType(B, A) == graph.head:

                        # R1 add non-collider
                        if graph.incomingArrowType(B, C) == graph.unknown:
                            uncertain_collider = False
                            if conservative:
                                uncertain_collider = B in uncertain_triples[(A,C)] or B in uncertain_triples[(C,A)]
                            if not (conservative and uncertain_collider):
                                graph.updateEdge(graph.head, C, B, rule)
                                graph.updateEdge(graph.tail, B, C, rule)
                                found_orientation = True

                        # R3 complete triple collider
                        elif graph.incomingArrowType(B, C) == graph.head:
                            for D in graph.adjacencies(B):
                                if graph.incomingArrowType(B, D) == graph.unknown:
                                    if A in graph.adjacencies(D) and graph.incomingArrowType(D, A) == graph.unknown:
                                        if C in graph.adjacencies(D) and graph.incomingArrowType(D, C) == graph.unknown:
                                            graph.updateEdge(graph.head, B, D, rule)
                                            found_orientation = True


def completeUndirectedEdges(graph):
    """
        Complete undirected edges when we assume that there might be selection bias.
            :param graph: CausalGraph
    """

    found_orientation = True
    while found_orientation:
        found_orientation = False

        # R5
        for A in graph.variables:
            circle_adjacencies = []
            for variable in graph.adjacencies(A):
                if graph.incomingArrowType(A, variable) == graph.unknown and graph.incomingArrowType(variable, A) == graph.unknown:
                    circle_adjacencies.append(variable)
            circle_adjacency_pairs = permutations(circle_adjacencies, 2)
            for B, C in circle_adjacency_pairs:
                if B not in graph.adjacencies(C):
                    for D in graph.adjacencies(B):
                        if A != D and D not in graph.adjacencies(A):
                            if graph.incomingArrowType(D, B) == graph.unknown and graph.incomingArrowType(B, D) == graph.unknown:
                                uncovered_circle_paths = findUncoveredCirclePath(graph, A, B, C, D)
                                for path in uncovered_circle_paths:
                                    for i in range(len(path)-1):
                                        graph.updateEdge(graph.tail, path[i], path[i+1])
                                        graph.updateEdge(graph.tail, path[i+1], path[i])
                                if uncovered_circle_paths != []:
                                    graph.updateEdge(graph.tail, B, A)
                                    graph.updateEdge(graph.tail, A, B)
                                    found_orientation = True

        # R6 and R7
        for B in graph.variables:
            adjacency_pairs = permutations(graph.adjacencies(B), 2)
            for A, C in adjacency_pairs:
                if graph.incomingArrowType(B, C) == graph.unknown and graph.incomingArrowType(A, B) == graph.tail:
                    if graph.incomingArrowType(B, A) == graph.tail:
                        graph.updateEdge(graph.tail, B, C)
                        found_orientation = True
                    elif graph.incomingArrowType(B, A) == graph.unknown and A not in graph.adjacencies(C):
                        graph.updateEdge(graph.tail, B, C)
                        found_orientation = True


def completeDirectedEdges(graph, CDC=False):
    """
    Rules 8-10 in Zhang (2008).
        :param graph: CausalGraph
        :param CDC: boolean (if CDC, track CDC orientations)
    """

    found_orientation = True

    if CDC:
        rule='FCI'
    else:
        rule=None

    while found_orientation:
        found_orientation = False

        for A in graph.variables:
            for C in graph.adjacencies(A):
                if graph.incomingArrowType(C, A) == graph.head and graph.incomingArrowType(A, C) == graph.unknown:

                    directed_edge = False

                    # R8
                    for B in graph.adjacencies(A):
                        if B in graph.adjacencies(C):
                            if graph.incomingArrowType(B, A) != graph.tail and graph.incomingArrowType(A, B) == graph.tail:
                                if graph.isDirectedEdge(B,C):
                                    directed_edge = True
                                    break

                    # R9
                    if not directed_edge:
                        uncovered_pd_paths = findUncoveredPotentiallyDirectedPath(graph, A, C)
                        for path in uncovered_pd_paths:
                            if path[1]!=C and path[1] not in graph.adjacencies(C):
                                directed_edge = True
                                break

                    # R10
                    if not directed_edge:
                        adjacencies = graph.adjacencies(C)
                        adjacencies.remove(A)
                        parents = [variable for variable in adjacencies if graph.isDirectedEdge(variable, C)]
                        parents_pairs = combinations(parents, 2)
                        for B, D in parents_pairs:
                            if directed_edge: break
                            AB_uncovered_pd_paths = findUncoveredPotentiallyDirectedPath(graph, A, B)
                            AD_uncovered_pd_paths = findUncoveredPotentiallyDirectedPath(graph, A, D)
                            for AB_path in AB_uncovered_pd_paths:
                                if directed_edge: break
                                for AD_path in AD_uncovered_pd_paths:
                                    if AB_path[1] != AD_path[1] and AB_path[1] not in graph.adjacencies(AD_path[1]):
                                        directed_edge = True
                                        break

                    if directed_edge:
                        graph.updateEdge(graph.tail, A, C, rule)
                        found_orientation = True

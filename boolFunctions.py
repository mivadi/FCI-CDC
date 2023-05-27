from causalGraph import CausalGraph


def isOpenRecursive(graph, current, previous, final, visited, ancestors_Z, Z, acyclic, SCCs):
    """
    Recursive function based on basebal search algorithm that returns a boolean
    indicating if there is an open path found given seperating set Z.
        :param graph: causalGraph object
        :param current, previous: current and previous vertex on a path.
        :param final: final vertex on the path
        :param visited: all visited vertices on the path
        :param ancestors_Z: list of ancestors for vertices Z
        :param Z: possible separating set (set of vertices)
        :param acylic: true if acyclic graph
        :param SCCs: dictionary that contains all strongly connected component info
    """
    # select new next vertices on the path
    nexts = [V for V in graph.adjacencies(current) if not V in visited] # this can faster
    open = False
    i = 0
    # once we find an open path, we know that we cannot block the vertices
    while not open and i < len(nexts):
        # check if the current triple form a collider
        collider = graph.incomingArrowType(current, previous)==graph.head and graph.incomingArrowType(current, nexts[i])==graph.head
        # path is not (yet) blocked when current is a collider and current is an ancestor of Z
        if collider and current in ancestors_Z:
            open = True
        # path is not (yet) blocked when current is a non-collider and current is not in Z
        elif not collider and not current in Z:
            open = True
        # for a cyclic graph we also check if the path is blocked according to sigma-separation
        elif not collider and not acyclic:
            if not ( graph.isDirectedEdge(current, nexts[i]) and nexts[i] not in SCCs[current] ):
                if not ( graph.isDirectedEdge(current, previous) and previous not in SCCs[current] ):
                    print("We have to test if it also works for cyclic graphs.")
                    open = True
        # if the path is open, we continue to the next triple on the path and test
        # again if the path is open for the next triple
        if open and nexts[i] != final:
            open = isOpenRecursive(graph, nexts[i], current, final, visited+[current], ancestors_Z, Z, acyclic, SCCs)

        i = i+1

    return open


def isSeparated(graph, X, Y, Z, ancestors, acyclic=True, SCCs=None):
    """
    Checks if X and Y are m-separated or sigma-separated by Z.
        :param graph: causalGraph object
        :param X, Y: starting and end vertex
        :param Z: possible separating set (set of vertices)
        :param ancestors: dictionary that contains all list of ancestors for each vertex
        :param acylic: true if acyclic graph
        :param SCCs: dictionary that contains all strongly connected component info
    """

    # select the ancestors for Z
    ancestors_Z = []
    for variable in Z:
        ancestors_Z = ancestors_Z + ancestors[variable]
    ancestors_Z = list(set(ancestors_Z))

    # select the adjacencies of X as 'current' vertices
    currents = graph.adjacencies(X)
    open = False
    i = 0

    # while we do not find an open path, keep trying to find one
    while not open and i < len(currents):
        open = isOpenRecursive(graph, currents[i], X, Y, [X], ancestors_Z, Z, acyclic, SCCs)
        i += 1

    # X and Y are separated by Z if there is no open path
    return not open


def inSepSet(X, separating_sets):
    """
    Checks if X is in a separating set.
        :param X: vertex
        :param separating_sets: dictionary of separating sets
    """
    for set in separating_sets:
        if X in set:
            return True
    return False

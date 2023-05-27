
import numpy as np

class CausalGraph(object):

    def __init__(self, variables, data=None, corr=None, data_size=-1, head='h', tail='t', non_collider=None, unknown=None, default_edge='t', score=False, context=[], track_directed_paths=False):
        """
        CausalGraph is a graph representing the causalities between the [variables].
            :param variables: list of possible variables (or vertices)
            :param data: dictionary of numpy arrays of shape (1,N) where N is number of data points
            :param head: string representing arrow head (default='h')
            :param tail: string representing arrow tail (default='t')
            :param non_collider: string representing definite non-collider (default=None)
            :param unknown: string representing unkown edge-mark (default=None)
            :param default_edge:
            :param score: not used yet
            :param context: list of vertices
            :param track_directed_paths: boolean, helpful for PC-algorithm
        """

        self.corr = corr
        self.data_size = data_size
        if not corr is None and data_size == -1:
            raise ValueError('Provide the number of data points.')

        self.data = data
        if data is not None:
            self.data_size = data[[*data][0]].shape[0]

        self.variables = variables
        self.context = context
        self.all_variables = variables + context

        self.default_edge = default_edge

        self.head = head
        self.tail = tail
        self.non_collider = non_collider
        self.unknown = unknown

        self.possible_types = [head, tail]
        if not self.unknown is None:
            self.possible_types.append(self.unknown)
        elif default_edge=='u':
            self.unknown = default_edge
            self.possible_types.append(self.unknown)

        # if non_collider is not None:
        #     self.possible_types.append(non_collider)
        #     self.def_non_colliders = True
        # else:
        #     self.def_non_colliders = False

        if unknown is not None:
            self.assign_unknown = True
            self.possible_types.append(unknown)
        else:
            self.assign_unknown = False

        self.track_directed_paths = track_directed_paths

        if self.track_directed_paths:
            self.directed_edges = set()
            self.directed_paths = set()


        # initialization of the complete undirected graph
        self.graph = {}
        for current in self.all_variables:
            self.graph[current] = {}
            for neighbour in self.all_variables:
                # no self-loops
                if current != neighbour:
                    # add edges with default incoming arrow types
                    # set default edge with non-collider on [] (list of all other vertices were it has to be a non-collider with) and score on 0
                    self.graph[current][neighbour] = [default_edge, [], 0]


    def do(self, variable):
        """
        Do-operator: removes all incoming edges of variable.
            :param variable: current variable
        """
        parents = self.parents(variable)
        for parent in parents:
            self.deleteEdge(variable, parent)


    def getData(self, variables):
        """
        Returns the data (length N) of the K variables: NxK array
            :param variables: subset (list) of variables from self.variables
        """
        if self.data is None:
            raise ValueError('Data is None.')

        if len(variables) > 0:

            vdata = self.data[variables[0]]

            if len(variables) > 1:
                for variable in variables[1:]:
                    vdata = np.append(vdata, self.data[variable], 1)
        else:
            vdata = None

        return vdata

    def resetAssignUnknown(self, bool):
        self.assign_unknown = bool

    def incomingArrowType(self, current, neighbour):
        """
        Returns the type of the arrow at [current] coming from [neighbour].
            :param current: variable from variables
            :param neighbour: variable from variables
        """
        return self.graph[current][neighbour][0]

    def isDefiniteNonCollider(self, neighbour1, current, neighbour2):
        """
        Returns a boolean if there is a definite non-collider at current between neighbour1 and neighbour2
            :param neighbour1/current/neighbour2: variable in variables
        """

        if self.incomingArrowType(current, neighbour1) == self.tail or self.incomingArrowType(current, neighbour2) == self.tail:
            return True
        if neighbour2 in self.graph[current][neighbour1][1] and neighbour1 in self.graph[current][neighbour2][1]:
            return True
        else:
            return False

    def score(self, current, neighbour):
        """
        Returns the score of the arrow at [current] coming from [neighbour].
            :param current/neighbour: variable from variables
        """
        return self.graph[current][neighbour][2]


    def isCollider(self, neighbour1, current, neighbour2):
        if self.incomingArrowType(current, neighbour1) == self.head and self.incomingArrowType(current, neighbour2) == self.head:
            return True
        else:
            return False


    def adjacencies(self, variable):
        """
        Returns the adjacencies of [variable].
            :param variable: variable from variables
        """
        return list(self.graph[variable].keys())

    def parents(self, variable):
        """
        Returns the parents of [variable].
            :param variable: variable from variables
        """
        parents = []
        for neighbour in self.graph[variable].keys():
            if self.isDirectedEdge(neighbour, variable):
                parents.append(neighbour)
        return parents

    def print(self, variable):
        """
        Prints the adjacency variables of [variable] in the graph.
            :param variable: variable from variables
        """
        print(self.graph[variable])

    def isDirectedEdge(self, variable1, variable2):
        """
        Returns a boolean: True if and only if there is a directed edge from
        variable1 to variable2.
        """
        if self.incomingArrowType(variable1, variable2) == self.tail and self.incomingArrowType(variable2, variable1) == self.head:
            return True
        else:
            return False

    def addEdge(self, variable1, variable2, type1, type2, dnc1=[], dnc2=[], score1=0, score2=0):
        """
        Add a (new) edge: (variable1 type1-type2 variable2).
            variable1: variable from variables
            variable2: variable from variables
            type1:   type of edge on side of variable1
            type2:   type of edge on side of variable2
        """
        self.isCorrectType(type1)
        self.isCorrectType(type2)
        self.graph[variable1][variable2] = (type1, dnc1, score1)
        self.graph[variable2][variable1] = (type2, dnc2, score2)

        if self.track_directed_paths:
            if type1==self.head and type2==self.tail:
                self.updateDirectedEdgesAndPaths((variable2,variable1))
            elif type1==self.tail and type2==self.head:
                self.updateDirectedEdgesAndPaths((variable1,variable2))


    def updateEdge(self, type, current, neighbour, rule=None):
        """
        Update an existing edge.
            :param type:   type of edge (string); 't':tail or 'h':head
            :param current: variable from variables
            :param neighbour: variable from variables
            :param rule: CDC orientation rule (default=None)
        """
        self.isCorrectType(type)
        self.isWellDefinedEdge(current, neighbour)
        self.isDefinedEdge(current, neighbour)

        # update the edge in case it is currently unknown
        # otherwise, update it anyway
        if (self.assign_unknown and self.graph[current][neighbour][0]==self.unknown) or not self.assign_unknown:

            # update orientation track matrices for CDC
            if self.graph[current][neighbour][0]!=type:
                i = self.all_variables.index(neighbour)
                j = self.all_variables.index(current)
                if rule == 'CDC':
                    self.CDC_orientations[i,j] = self.possible_types.index(type) + 2
                elif rule == 'parent':
                    self.parent_orientations[i,j] = self.possible_types.index(type) + 2
                elif rule == 'FCI':
                    self.FCI_orientations[i,j] = self.possible_types.index(type) + 2

            if rule == 'parent':
                self.parent_orientations_all[i,j] = self.possible_types.index(type) + 2

            # update arrow type
            self.graph[current][neighbour][0] = type


        # in case we track the directed edges and paths, we have to update them
            if self.track_directed_paths:
                if self.incomingArrowType(current, neighbour)==self.head and self.incomingArrowType(neighbour, current)==self.tail:
                    self.updateDirectedEdgesAndPaths((neighbour,current))
                elif self.incomingArrowType(current, neighbour)==self.tail and self.incomingArrowType(neighbour, current)==self.head:
                    self.updateDirectedEdgesAndPaths((current,neighbour))


    def updateCause(self, A, B, rule=None):
        """
        Update cause A->B.
            :param A: variable from variables
            :param B: variable from variables
            :param rule: CDC orientation rule (default=None)
        """
        self.updateEdge(self.head, B, A, rule)
        self.updateEdge(self.tail, A, B, rule)


    def deleteEdge(self, variable1, variable2):
        """
        Delete an edge from the graph.
            :param variable1: variable from variables
            :param variable2: variable from variables
        """
        if variable1 in self.graph[variable2] and variable2 in self.graph[variable1]:
            del self.graph[variable1][variable2]
            del self.graph[variable2][variable1]
        else:
            raise ValueError("No such edge in graph.")


    def directedEdges(self):
        """
        Returns all directed edges.
        """
        if not self.track_directed_paths:
            raise TypeError("Directed edges are not tracked.")
        return list(self.directed_edges)


    def directedPaths(self):
        """
        Returns all directed paths.
        """
        if not self.track_directed_paths:
            raise TypeError("Directed paths are not tracked.")
        return list(self.directed_paths)


    def updateDirectedEdgesAndPaths(self, edge):
        """
        Updates the sets of directed edges and paths.
        Note that tracking works in case of PC algorithm.
        Due to reassigning and causal insufficienty, it does not work for FCI.
            :param edge: ordered tuple (X,Y) so that X->Y is a directed edge in the graph
        """

        if not self.track_directed_paths:
            raise TypeError("Directed edges and paths are not tracked.")

        self.directed_edges.add(edge)

        self.directed_paths.add(edge)

        done = False
        while not done:
            done = True
            new_directed_paths = []
            # loop over all pairs X->Y1 and Y2->Z
            for X, Y1 in self.directed_paths:
                for Y2, Z in self.directed_paths:
                    # in case the end vertex of first path is the vertex of the second paths
                    # we find a new path and we will add it to the path set
                    if Y1 == Y2 and (X,Z) not in self.directed_paths:
                        new_directed_paths.append((X,Z))
                        done = False
            self.directed_paths.update(new_directed_paths)


    def getCDCorientationMatrices(self):
        """
        Intializes adjacency matrices for the orientation rules in the CDC phase.
        """
        len_vars = len(self.all_variables)
        mat = np.zeros((len_vars, len_vars))
        self.CDC_orientations = np.zeros((len_vars, len_vars))
        self.parent_orientations = np.zeros((len_vars, len_vars))
        self.parent_orientations_all = np.zeros((len_vars, len_vars))
        self.FCI_orientations = np.zeros((len_vars, len_vars))


    def adjacencyMatrix(self):
        """
        Returns adjacency matrix.
        """
        len_vars = len(self.all_variables)
        adj_mat = np.zeros((len_vars, len_vars))

        for i in range(len_vars):
            for j in range(len_vars):
                # check if adjacent
                if self.all_variables[j] in self.adjacencies(self.all_variables[i]):
                    # check what kind of edge-mark we have here
                    incoming_arrow_type = self.incomingArrowType(self.all_variables[j], self.all_variables[i])
                    if self.unknown is not None and incoming_arrow_type == self.unknown:
                        adj_mat[i,j] = 1 # i*-oj
                    elif incoming_arrow_type == self.head:
                        adj_mat[i,j] = 2 # i*->j
                    elif incoming_arrow_type == self.tail:
                        adj_mat[i,j] = 3 # i*-j
        return adj_mat


    def isCorrectType(self, type):
        """
        Checks if the type of edge end is correct.
            :param type:   type of edge (string); 't':tail or 'h':head
        """
        if type not in self.possible_types:
            raise ValueError("Type {} is not known. Choose type from the list {}.".format(type, self.possible_types))


    def isWellDefinedEdge(self, variable1, variable2):
        """
        Checks if the edge is well-defined, in other words,
        check if the edge is (not) saved in the dictionary of both variables.
            :param variable1: variable from variables
            :param variable2: variable from variables
        """
        # check for XOR
        if variable1 not in self.graph[variable2] != variable2 not in self.graph[variable1]:
            raise ValueError("Edge between {} and {} is not well defined.".format(variable1, variable2))


    def isDefinedEdge(self, variable1, variable2):
        """
        Checks if the edge exists.
            :param variable1: variable from variables
            :param variable2: variable from variables
        """
        # check for AND
        if variable1 not in self.graph[variable2] and variable2 not in self.graph[variable1]:
            raise ValueError("Edge between {} and {} does not exist.".format(variable1, variable2))

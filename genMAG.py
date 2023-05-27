from causalGraph import CausalGraph
from searchFunctions import findAncestors
from boolFunctions import isSeparated
import numpy as np
import random
from scipy.linalg import expm


def inv_perm(perm):
    """
    Returns inverse of permutation perm.
        :param perm: index list containing 0,1,2,...,len(perm)
    """
    inverse = [0] * len(perm)
    try:
        for i, p in enumerate(perm):
            inverse[p] = i
    except:
        raise ValueError('perm should contain integers 0,1,2,...,len(perm)')
    return inverse


# ======================================================================
def AddBi(x,y, M, A):
    """
    Add bi-directed edge x <-> y to M, verify valid, and update A.
        :param x, y: indices corresponding to vertices 
        :param M: adjacency matrix - numpy array of dimension (N,N)
        :param A: ancestral graph matrix - numpy array of dimension (N,N)
    """

    # update adjacency matrix
    M[x,y] = 2
    M[y,x] = 2

    # verify if update of ancestral matrix is possible
    if A[x,y] == 1 or A[y,x] == 1:
        print(f'ERROR - AddBi: {x} <-> {y}\n')

    # update ancestral matrix
    A[x,y] = 2
    A[y,x] = 2


def AddArc(x, y, M, A ):
    """
    Add arc x --> y to M, verify valid, and update A.
        :param x, y: indices corresponding to vertices 
        :param M: adjacency matrix - numpy array of dimension (N,N)
        :param A: ancestral graph matrix - numpy array of dimension (N,N)
    """

    # update adjacency matrix
    M[x,y] = 1
    M[y,x] = 2
    
    # select ancestors x
    anX = np.where(A[:,x] == 1)[0]

    # select descendants y
    deY = np.where(A[y,:] == 1)[0]

    # check if valid causal relationship
    if np.any(A[anX[:, None],deY] == 2) or np.any(A[deY[:, None],anX]==1):
        print(f'ERROR - AddArc: {x} --> {y}\n')
    
    # update ancestral matrix for all AnX -> DeY
    A[anX[:, None],deY] = 1
    A[deY[:, None],anX] = 2


def myunion(A,B):
    """
    Returns the union of A and B.
        :param A, B: list of vertices
    """

    # copy the list
    C = list(A) 

    # take union of list
    C.extend(B) 

    # delete doubles
    return list(set(C)) 


def myintersect(A,B):
    """
    Returns the intersection of A and B.
        :param A, B: list of vertices
    """

    # copy the list
    C = list(A) 

    # take intersection
    return list(set(C).intersection(B))


def reachability_graph(G, self_loop=0):
    """
    Returns the reachability graph C for which C(i,j)=1 iff 
    there is a path from i to j in DAG G.
        :param G: ancestral matrix
        :param self_loop: integer 0 or 1 indicating if we in/exclude self-loops
    """

    # based on BayesNetToolbox\FullBNT-1.0.4\graph\reachability_graph.m
    # tipping point around n = 14
    n = len(G)

    # for MAG/PAG: remove arrowheads (== 2) and set circle (==3) to tail
    G = np.copy(G)
    G[G == 2] = 0
    G[G == 3] = 1

    # check possible directed paths
    if (n > 14):

        # expm(G) = I + G + G^2 / 2! + G^3 / 3! + ...
        if self_loop == 0:
            M = expm(G) - np.eye(n) # exclude self (only proper ancestors)
        else:
            M = expm(G) # do not exclude self : x \in An(x)
        C = M > 0

    else:

        # This computes C = G + G^2 + ... + G^{n-1} but include self for acyclic
        A = np.copy(G)
        if self_loop == 0:
            C = np.zeros((n,n)) # exclude self (default)
        else:
            C = np.eye(n) # include self

        for i in range(n):
            C = C + A
            A = A @ G
        C = C > 0

    return C


def mk_ag(N, maxK=5, avgK=None, p_bi=0.2, p_sel=0):
    """
    Returns a random ancestral graph over N variables with max node
    degree maxK, and probability on bi-directed edge = p_bi. 
    First we generate undirected skeleton over N variables by starting from empty graph,
    and then adding random edges. Then we pick an edge and choose random orientation.
        :param N: number of nodes
        :param maxK: max node degree
        :param avgK: average node degree (2 < avgK < maxK)
        :param p_bi: probability of edge being bidirected
        :param p_sel: probability of selection bias per node (default 1/N)
    """

    # convert N, maxK into min/max/avg edge limits
    posEdge = N*(N-1)/2
    minEdge = (N-1)
    maxEdge = N*maxK/2

    # check edge probabilities, if necessary adjust p_Edge to obtain
    # connected but not too dense graphs
    if avgK is None:
        avgK = (1 + maxK/2)
    elif avgK < 2:
        avgK = 2
    elif avgK > maxK:
        avgK = maxK
    p_Edge = avgK/(N-1)

    # sample number of edges until nEdge falls within valid interval
    # NOTE: not guaranteed exactly nEdge edges in M. If close to maximum for
    # given maxK, then in some configurations no more edges can be added.
    nEdge = 0
    while nEdge < minEdge or nEdge > maxEdge:
        nEdge = np.random.binomial(posEdge, p_Edge)

    # initialize empty graph
    M = np.zeros((N,N))

    # initialize zero degree for all nodes
    D = np.zeros(N)

    # initialize array of nodes not yet connected (prio)
    V0 = list(np.arange(N))

    # initialize array of nodes to choose from (prio 2)
    Vk = []

    # initialize array of nodes with degree = maxK (no more candidate)
    VK = []

    # created connected skeleton over nEdge edges
    # start with random node in Vk (and remove from V0)
    x = np.random.randint(N)
    Vk.append(x)
    V0.remove(x)

    # first connect all nodes to obtain a connected component
    for nE in range(N-1):

        # start with one from Vk then one from V0 or Vk
        x = np.random.choice(Vk)

        # new not-yet-connected node
        y = np.random.choice(V0)

        # add undirected edge to M
        M[x,y] = 1
        M[y,x] = 1

        # update arrays/degrees
        Vk.append(y)
        V0.remove(y)

        # increase degree of both
        D[[x,y]] = D[[x,y]]+1

        # move nodes at max.degree to VK (maxK > 1, so never Vj)
        if D[x] >= maxK:
            VK.append(x)
            Vk.remove(x)


    if len(V0)>0: raise ValueError('Error mk_ag.1')

    # now add all other edges
    for nE in range(N, nEdge+1):

        # sample node from Vk 
        x = np.random.choice(Vk)

        # copy Vk to select candidate neighbours
        Vx = list(Vk)

        for y in Vk:
            # remove invalid candidates: no self-loops or already existing edges
            if x == y or M[x,y] > 0:
                Vx.remove(y)

        # it is possible that Vx becomes empty, so then just continue
        if len(Vx)==0: continue

        # sample node from Vx
        y = np.random.choice(Vx)

        # add undirected edge to M
        M[x,y] = 1
        M[y,x] = 1

        # increase degree of both
        D[[x,y]] = D[[x,y]]+1

        # check degree, first y / j
        if D[y] >= maxK:
            VK.append(y)
            Vk.remove(y)

        # check degree, then x / i
        if D[x] >= maxK:
            VK.append(x)
            Vk.remove(x)

    # change undirected edges to circle marks
    M = 3 * M

    # set random time-order to ensure consistent orientations for arcs
    Vt = list(np.random.permutation(N))

    # get inverse permutation
    Vinvt = inv_perm(Vt)

    # ancestor graph where A(x,y)=2 means x not ancestor of y
    # and A(x,y)=1 means x is ancestor of y
    A = np.tril(2 * np.ones((N,N)) - np.eye(N))
    A = A[np.array(Vinvt)[:, None], np.array(Vinvt)]

    # set counters
    nArc = 0 # number of arcs allocated in M
    nBi  = 0 # number of bi-directed arcs in M

    # per node: orientations to do is degree unoriented
    Dx = np.copy(D) # number of edges at node to orient

    # loop until all edges are oriented
    while np.sum(Dx) > 0:

        # select node with unoriented edge
        Vx = np.where(Dx > 0)[0]
        x = np.random.choice(Vx)

        # select one from set of nodes on unoriented edge to x
        Vy = np.where(M[x,:] == 3)[0]
        y = np.random.choice(Vy)

        # ensure x->y consistent with Vt
        if Vinvt[x] > Vinvt[y]:
            # swap direction
            tmp = x
            x = y
            y = tmp

        # update number of unoriented edges
        Dx[[x,y]] = Dx[[x,y]] - 1

        # decide if we will add bidirected edge or arc
        add_bi = random.uniform(0, 1) < p_bi

        if add_bi:
            # add bidirected edge
            AddBi(x, y, M, A)
            nBi += 1
        else:
            # add arc x->y
            AddArc(x, y, M, A)
            nArc = nArc + 1

        # process
        if add_bi:
            # avoid almost directed cycles by orienting all unoriented 
            # edges De(x) o-o An(y) as bidirected

            # select descendants and ancestors
            DeX = np.where(A[x,:] == 1)[0]
            AnY = np.where(A[:,y] == 1)[0]

            # find and process all unoriented edges De(x)-An(y)
            matches = [(u,v) for u in DeX for v in AnY]
            for u, v in matches:
                if M[u,v]==3:

                    # if arc would be u -> v, then a.d.cycle
                    if Vinvt[u] < Vinvt[v]:

                        # add bi-directed edge
                        AddBi(u, v, M, A)
                        nBi += 1

                        # update edges-to-orient counter
                        Dx[[u,v]] = Dx[[u,v]] - 1

        else:
            
            # select descendants and ancestors
            AnX = np.where(A[:,x] == 1)[0]
            DeY = np.where(A[y,:] == 1)[0]

            # avoid cycles by orienting all unoriented edges An(x) o-o De(y) as An(x) -> De(y)
            matches = [(u,v) for u in AnX for v in DeY]
            for u, v in matches:
                if M[u,v]==3:

                    # add arc u -> v
                    AddArc(u,v, M, A)
                    nArc = nArc + 1

                    # update edges-to-orient counter
                    Dx[[u,v]] = Dx[[u,v]] - 1

            # for each spouse z of An(x), test for the ancestors of z An(z):
            # if the edge AnZu - DeYv is unoriented, and if AnZu > DeYv in Vt,
            # then add AnZu <-> DeYv
            for a in AnX:

                # loop over all spouses of the ancestors of x
                Zs = np.intersect1d(np.where(M[a,:] == 2)[0], np.where(M[:,a] == 2)[0])
                for z in Zs:

                    # select the ancestors of z
                    AnZ = np.where(A[:,z] == 1)[0]

                    # now find unoriented edges AnZu - DeYv
                    matches = [(u,v) for u in AnZ for v in DeY]
                    for u, v in matches:
                        if M[u,v]==3:

                            # check v << u in Vt
                            if Vinvt[u] > Vinvt[v]:

                                # add bi-dir u <-> v
                                AddBi(u, v, M, A)
                                nBi += 1

                                # update edges-to-orient counter
                                Dx[[u,v]] = Dx[[u,v]] - 1

            # for each spouse z of De(y), test for the descendants of z De(z),
            # if the edge dez_j - anx_k is unoriented, and dez_j < anx_k in Vt,
            # then add anz_j <-> dey_k
            for d in DeY:

                # loop over all spouses of the decendants of y
                Zs = np.intersect1d(np.where(M[d,:] == 2)[0], np.where(M[:,d] == 2)[0])
                for z in Zs:

                    # select the decendants of z
                    DeZ = np.where(A[z,:] == 1)[0]

                    # now find unoriented edges AnZ - DeY
                    matches = [(u,v) for u in DeZ for v in AnX]
                    for u, v in matches:
                        if M[u,v]==3:

                            # check v << u in Vt
                            if Vinvt[u] < Vinvt[v]:

                                # add bi-dir u <-> v
                                AddBi(u, v, M, A)
                                nBi += 1

                                # update edges-to-orient counter
                                Dx[[u,v]] = Dx[[u,v]] - 1

    # verify if M is an ancestral graph
    debug=1
    if debug > 1:

        # no unprocessed edges
        if np.any(M > 2):
            print(f'ERROR - unprocessed edges')

        # get reachability graph R2 from M
        B = np.copy(M)
        # remove arrowheads to obtain adjacency matrix containing the directed edges only
        B[B == 2] = 0 
        Bn = np.copy(B)
        R2 = np.zeros((N,N))
        # Taylor series trick to count self-cycles
        for i in range(N):
            R2 = R2 + Bn
            Bn = Bn @ B
        R2[R2 > 0] = 1

        # check for self-loops
        if np.any(np.diagonal(R2)>0):
            print(f'ERROR - self loop in M')

        # obtain all bi-directed edges
        B = np.triu(M + M.T)
        X, Y = np.where(B == 4)

        # loop over all bi-directed edges to check if there is an almost directed cycle
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            if R2[x,y] > 0 or R2[y,x] > 0:
                print(f'ERROR - almost directed cycle {x} <-> {y}\n')

    return M


def adjmatrix2CausalGraph(M):
    """
    Returns CausalGraph object corresponding to M and the index to variable function.
        :param M graph (adjacency matrix)
    """

    # create index to variable function
    variables = []
    i2v = {}
    for i in range(len(M)):
        variables.append('v'+str(i))
        i2v[i] = variables[-1]

    # create corresponding CausalGraph
    G = CausalGraph(variables, unknown='u', default_edge='u')
    for i in range(M.shape[0]):

        for j in range(M.shape[1]):

            if M[i,j] != 0  and not i2v[i] in G.adjacencies(i2v[j]):
                raise ValueError('Inconsistant adjacency matrix.')

            if i2v[i] in G.adjacencies(i2v[j]):

                if M[i,j] == 0:
                    # remove edge
                    G.deleteEdge(i2v[i], i2v[j])

                elif M[i,j] == 1:
                    # add tail: i -* j
                    G.updateEdge(G.tail, i2v[i], i2v[j])

                elif M[i,j] == 2:
                    # add head: i <-* j
                    G.updateEdge(G.head, i2v[i], i2v[j])

    return G, i2v


def ag_to_mag(G):
    """
    Returns adjacency matrix and CausalGraph object of the maximal 
    ancestral graph obtained from an ancestral graph.
        :param G: ancestral graph (adjacency matrix)
    """

    # initialize
    N = len(G)
    M = np.copy(G)
    R = reachability_graph(G)
    nIndEdge = 0

    # get CausalGraph object and index to variable function
    AG, i2v = adjmatrix2CausalGraph(M)

    # find ancestors
    ancestors = findAncestors(AG)

    # loop over all combinations and test for m-separation given ancestors
    for x in range(N-1):
        for y in range(x+1, N):

            # skip nodes already adjacent in G
            if G[x,y] > 0: continue

            # find all ancestors of x and/or y and/or S
            AxyS = np.where((np.sum(R[:,[x,y]], axis=1) > 0 ))[0]
            AxyS = [i2v[z] for z in AxyS[(AxyS!=x) & (AxyS!=y)]]

            # test for m-separation given all ancestors
            if not isSeparated(AG, i2v[x], i2v[y], AxyS, ancestors):

                # inducing pathfrom x to y: not separable, so add edge

                if R[x,y] > 0:
                    M[x,y] = 1
                else:
                    M[x,y] = 2

                if R[y,x] > 0:
                    M[y,x] = 1
                else:
                    M[y,x] = 2

                nIndEdge = nIndEdge + 1

    # get CausalGraph object of MAG
    MAG, _ = adjmatrix2CausalGraph(M)

    return M, MAG


def mk_mag(n, p_bi=.2):
    """
    Returns a random maximal ancestral graph over n variables with probability on bi-directed edge is p_bi. 
        :param n: number of nodes
        :param p_bi: probability of edge being bidirected
    """

    # generate random ancenstral graph
    AG = mk_ag(n, p_bi=p_bi)

    # get maximal ancestral graph obtained from the ancestral graph
    M, MAG = ag_to_mag(AG)

    # return the CausalGraph object
    return MAG

from causalGraph import CausalGraph


def TarjansAlgorithm(graph, acylic=False):
    """
    Algorithm to find strongly connected components: "Depth-first search and linear graph algorithms" - Tarjan (1972).
    """
    vertices = {}
    i = 0
    SCCs = {}
    causes = {}
    for v in graph.all_variables:
        if v not in vertices.keys():
            vertices, i, SCCs = strongConnect(graph, vertices, v, i, SCCs, causes)

    for v in vertices.keys():
        if vertices[v]['onStack']:
            causes[vertices[v]['index']] = vertices[v]['causes']

    latest = latestCauses(causes)
    ordered_SCCs = []
    while len(latest)>0:
        for j in latest:
            ordered_SCCs = [SCCs[j]] + ordered_SCCs
            for i in causes.keys():
                for v in SCCs[j]:
                    if vertices[v]['index'] in causes[i]:
                        causes[i].remove(vertices[v]['index'])
        latest = latestCauses(causes)

    return ordered_SCCs


def latestCauses(causes):
    latest = []
    for j in causes.keys():
        if len(causes[j])==0:
            latest.append(j)
    for j in latest:
        del causes[j]
    return latest


def strongConnect(graph, vertices, v, i, SCCs, causes):
    vertices[v] = {}
    vertices[v]['index'] = i
    vertices[v]['lowlink'] = i
    vertices[v]['onStack'] =  True
    SCCs[i] = [v]
    vertices[v]['causes'] = []
    i+=1
    adjacencies = graph.adjacencies(v)

    for w in adjacencies:

        if graph.isDirectedEdge(v, w):

            if w not in vertices.keys():
                vertices, i, SCCs = strongConnect(graph, vertices, w, i, SCCs, causes)
                vertices[v]['lowlink'] = min(vertices[v]['lowlink'], vertices[w]['lowlink'])
            elif vertices[v]['onStack']:
                vertices[v]['lowlink'] = min(vertices[v]['lowlink'], vertices[w]['lowlink'])

            # track causes
            vertices[v]['causes'].append(vertices[w]['index'])

    if vertices[v]['lowlink'] == vertices[v]['index']:

        causes[vertices[v]['lowlink']] = vertices[v]['causes']
        vertices[v]['onStack'] =  False

        for w in vertices:
            if v != w:
                if vertices[w]['lowlink'] == vertices[v]['lowlink']:

                    vertices[w]['onStack'] =  False

                    # add to SCC of v
                    SCCs[vertices[v]['index']].append(w)

                    # add causes
                    causes[vertices[v]['lowlink']] = causes[vertices[v]['lowlink']] + vertices[w]['causes']

                    # delete from own SCC
                    del SCCs[vertices[w]['index']]

        causes[vertices[v]['lowlink']] = list(set(causes[vertices[v]['lowlink']]))
        for w in SCCs[vertices[v]['index']]:
            if vertices[w]['index'] in causes[vertices[v]['lowlink']]:
                causes[vertices[v]['lowlink']].remove(vertices[w]['index'])

    return vertices, i, SCCs
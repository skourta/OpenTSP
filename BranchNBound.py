import numpy as np


def reduce_matrix(data):
    reduced = np.copy(data)
    rowMin = np.array([np.min(data[i, :]) for i in range(data.shape[0])])
    rowMin = rowMin[:, np.newaxis]
    rowMin[rowMin == np.inf] = 0
    # print(rowMin)
    reduced = data - rowMin
    colMin = np.array([np.min(reduced[:, i]) for i in range(reduced.shape[0])])
    colMin[colMin == np.inf] = 0
    # print(colMin)
    reduced = reduced - colMin
    # print(np.sum(rowMin)+np.sum(colMin))
    return reduced, np.sum(rowMin) + np.sum(colMin)


def getNextNode(fullNode, visited, active, start, nodeSuiv, size):

    data = fullNode[1]['matrix']
    node = fullNode[1]['node']
    cost = fullNode[1]['cost']
    # for oldNode in active:
    #     if(oldNode[0] == node and oldNode[1]['cost'] == )
    # print(visited)
    for i in range(0, data.shape[0]):
        if (not (i in visited)):
            temp = np.copy(data)
            temp[node, :] = np.inf
            temp[:, i] = np.inf
            # temp[i, node] = np.inf
            temp[i, start] = np.inf
            # print(temp,"\n")
            childReduced, childCost = reduce_matrix(temp)
            # print(childReduced)
            # print('> childCost',childCost,'edge cost',data[node,i],'parent cost',cost,'parent',node,'child',i,'<')
            childCost += data[node, i] + cost
            active.append((nodeSuiv, {'node': i, 'cost': childCost, 'matrix': np.copy(childReduced)}))
            nodeSuiv += 1
    seq = [x[1]['cost'] for x in active]
    minim = min(seq)
    index = -1
    if len(active) > size-1:
        for j in range(len(seq) - 1, 0, -1):
            if seq[j] == minim:
                index = j
                break
    else:
        index = seq.index(minim)
    return active.pop(index), nodeSuiv


def branchNbound(start, data):
    root = start
    visited = []
    active = []
    nodeSuiv = 0
    reduced, cost = reduce_matrix(np.copy(data))
    visited.append(root)
    nextNode = (nodeSuiv, {'node': start, 'cost': cost, 'matrix': np.copy(reduced)})
    nodeSuiv += 1
    nextNode, nodeSuiv = getNextNode(nextNode, visited, active, start, nodeSuiv,data.shape[0])
    while (len(visited) < data.shape[0]):
        visited.append(nextNode[1]['node'])
        prevCost = nextNode[1]['cost']
        nextNode, nodeSuiv = getNextNode(nextNode, visited, active, start, nodeSuiv,data.shape[0])
    summ = 0
    for i in range(len(visited) - 1):
        summ += (data[visited[i], visited[i + 1]])
    summ += data[visited[len(visited) - 1], start]
    # print(active)
    return visited, summ

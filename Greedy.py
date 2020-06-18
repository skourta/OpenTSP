import numpy as np
import Parser
import argparse
import time
from array import array as pyarray


def restore_path(connections, endpoints):
    # Takes array of connections and returns a path.

    if endpoints is None:
        start, end = [idx
                      for idx, conn in enumerate(connections)
                      if len(conn) == 1]
    else:
        start, end = endpoints

    path = [start]
    prev_point = None
    cur_point = start
    while True:
        next_points = [pnt for pnt in connections[cur_point]
                       if pnt != prev_point]
        if not next_points: break
        next_point = next_points[0]
        path.append(next_point)
        prev_point, cur_point = cur_point, next_point
    return path


def pairs_by_dist(N, distances):
    # returns list of coordinate pairs (i,j), sorted by distances; such that i < j
    indices = []
    for i in range(N):
        for j in range(i):
            indices.append(i * N + j)

    indices.sort(key=lambda ij: distances[ij // N][ij % N])
    return ((ij // N, ij % N) for ij in indices)


def path_cost(distance_matrix, path):
    # Caclulate total length of the given path, using the provided distance matrix.

    ipath = iter(path)
    T = len(path)
    start = path[0]
    end = path[T - 1]
    try:
        j = next(ipath)
    except StopIteration:
        # empty path
        return 0

    dist = distance_matrix[start][end]
    for i in ipath:
        if i >= j:
            dist += distance_matrix[i][j]
        else:
            dist += distance_matrix[j][i]
        j = i
    return dist


def solve_tsp_glouton(distances, endpoints=None):
    N = len(distances)
    if N == 0: return []
    if N == 1: return [0]

    # State of the TSP solver algorithm.
    node_valency = pyarray('i', [2]) * N

    if endpoints is not None:
        start, end = endpoints
        if start == end: raise ValueError("start=end is not supported")
        node_valency[start] = 1
        node_valency[end] = 1

    # for each node, stores 1 or 2 connected nodes
    connections = [[] for i in range(N)]

    def join_segments(sorted_pairs):
        # segments of nodes. Initially, each segment contains only 1 node
        segments = [[i] for i in range(N)]

        def possible_edges():
            for ij in sorted_pairs:
                i, j = ij
                if node_valency[i] and node_valency[j] and (segments[i] is not segments[j]):
                    yield ij

        def connect_vertices(i, j):
            node_valency[i] -= 1
            node_valency[j] -= 1
            connections[i].append(j)
            connections[j].append(i)
            # Merge segment J into segment I.
            seg_i = segments[i]
            seg_j = segments[j]
            if len(seg_j) > len(seg_i):
                seg_i, seg_j = seg_j, seg_i
                i, j = j, i
            for node_idx in seg_j:
                segments[node_idx] = seg_i
            seg_i.extend(seg_j)

        def edge_connects_endpoint_segments(i, j):
            # return True, if given ede merges 2 segments that have endpoints in them
            si, sj = segments[i], segments[j]
            ss, se = segments[start], segments[end]
            return (si is ss) and (sj is se) or (sj is ss) and (si is se)

        # Take first N-1 possible edge. they are already sorted by distance
        edges_left = N - 1
        for i, j in possible_edges():
            if endpoints and edges_left != 1 and edge_connects_endpoint_segments(i, j):
                continue  # don't allow premature path termination

            connect_vertices(i, j)
            edges_left -= 1
            if edges_left == 0:
                break

    # invoke main greedy algorithm
    join_segments(pairs_by_dist(N, distances))
    P = restore_path(connections, endpoints=endpoints)
    return P, path_cost(distances, P)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--start",
                        help="Starting Node", )
    args = parser.parse_args()
    instance = Parser.TSPInstance(args.instance)
    instance.readData()
    start_time = time.time()
    if args.start is not None:
        result = solve_tsp_glouton(np.array(instance.data), endpoints=(int(args.start), None))
    else:
        result = solve_tsp_glouton(np.array(instance.data), endpoints=None)
    end_time = time.time()
    tour = result[0]
    cost = result[1]
    print(tour)
    print(cost)
    print(end_time - start_time)

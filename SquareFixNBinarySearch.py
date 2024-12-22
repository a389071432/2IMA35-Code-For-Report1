import math
from copy import deepcopy
from datetime import datetime
from pyspark import SparkConf, SparkContext
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve
from sklearn.neighbors import KDTree
from argparse import ArgumentParser

from Plotter import *
from DataReader import *
from DataModifier import *
from MyDataPatterns import *


def get_clustering_data(ratio):
    bound = 0.1
    wL = (1 - bound)/(ratio+1)
    wR = ratio*(1 - bound)/(ratio+1)
    circles = final_make_pair_rectangle(
                                     {'N': 800, 'L': 0, 'R':wL },
                                     {'N': 800, 'L': 1-wR, 'R':1.0 }
                                    )

    dataset = circles
    return dataset


def map_contract_graph(_lambda, leader):
    def contraction(adj):
        u, nu = adj
        # c, v = u, u
        # S = []
        # while v not in S:
        #     S.append(v)
        #     c = min(c, v)
        #     v = _lambda[v]
        # c = min(c, v)
        c = leader[u]
        A = list(filter(lambda e: leader[e[0]] != c, nu))
        return c, A
    return contraction


def reduce_contract_graph(leader):
    def reduce_contraction(Nu, A):
        for v, w in A:
            l = leader[v]
            new = True
            for i, e in enumerate(Nu):
                if l == e[0]:
                    new = False
                    Nu[i] = (l, min(w, e[1]))
            if new:
                Nu.append((l, w))
        return Nu
    return reduce_contraction


# the heuristic: randomly drop the nearest edge at probability of p
def do_random_edge_adjustment(p):
    def random_edge_adjustment(adj):
        u, nu = adj
        if len(nu)>0:
           # Sort the list based on the second element (w) of each tuple
           sorted_nu = sorted(nu, key=lambda x: x[1])
    
           # Get the indices of the smallest and second smallest `w`
           smallest_idx = nu.index(sorted_nu[0])

           if random.random() < p:
              del nu[smallest_idx]
        return u, nu

    return random_edge_adjustment


def find_best_neighbours(adj):
    u, nu = adj
    nn = u

    if len(nu) > 0:
        min_v, min_w = nu[0]
        for v, w in nu:
            if w < min_w:
                min_v, min_w = v, w
        nn = min_v
    return u, nn


def find_leader(_lambda):
    def find(adj):
        u, nu = adj
        c, v = u, u
        S = []
        cnt = 0
        while v not in S:
            S.append(v)
            c = v
            v = _lambda[v]
            cnt += 1
        c = min(c, v)
        return u, c

    return find


def affinity_clustering(adj, vertex_coordinates, plot_intermediate, num_clusters=2, plotter=None):
    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext.getOrCreate(conf=conf)
    clusters = [[i] for i in range(len(adj))]
    yhats = []
    leaders = []
    graph = deepcopy(adj)
    rdd = sc.parallelize(adj)


    i = 0
    imax = 40
    contracted_leader = [None] * len(adj)
    mst = [None] * len(adj)
    while i < imax:
        if len(graph) <= num_clusters:
            break
        num_edges = sum(map(lambda v: len(v[1]), graph))
        if num_edges == 0:
            break

        rdd1 = rdd.map(find_best_neighbours).collect()
        _lambda = [None] * len(adj)
        for line in rdd1:
            _lambda[line[0]] = line[1]

        # Find leader
        leader = [None] * len(adj)
        rdd1 = rdd.map(find_leader(_lambda)).collect()
        for line in rdd1:
            leader[line[0]] = line[1]
        leaders.append(leader)


        for j in range(len(adj)):
            l = leader[j]
            if l is not None and not l == j:
                clusters[l].extend(clusters[j])
                clusters[j].clear()

        yhat = [None] * len(adj)
        for c, cluster in enumerate(clusters):
            for v in cluster:
                yhat[v] = c
        yhats.append(yhat)

        for j in range(len(adj)):
            if contracted_leader[j] is None:
                if yhat[j] != j:
                    contracted_leader[j] = yhat[j]
                    mst[j] = _lambda[j]

        # Contraction
        rdd = (rdd.map(map_contract_graph(_lambda=_lambda, leader=leader))
               .foldByKey([], reduce_contract_graph(leader)))

        graph = rdd.map(do_random_edge_adjustment(p=0.9)).collect()

        i += 1

    for j in range(len(adj)):
        if contracted_leader[j] is None:
            contracted_leader[j] = yhat[j]
            mst[j] = yhat[j]

    return i, graph, yhats, contracted_leader, mst, leaders


def get_nearest_neighbours(V, k=2, leaf_size=2, buckets=False):
    def get_sort_key(item):
        return item[1]

    V_copy = deepcopy(V)
    if buckets:
        adj = []
        for key in V:
            nu = []
            sorted_list = sorted(V_copy[key].items(), key=get_sort_key)
            last = -1
            to_shuffle = []
            for i in range(k):
                if last != sorted_list[i][1]:
                    to_shuffle.append((sorted_list[i][0], sorted_list[i][1]))
                    random.shuffle(to_shuffle)
                    for item in to_shuffle:
                        nu.append(item)
                    to_shuffle = []
                else:
                    to_shuffle.append((sorted_list[i][0], sorted_list[i][1]))
                last = sorted_list[i][1]

            random.shuffle(to_shuffle)
            for item in to_shuffle:
                nu.append(item)
            adj.append((key, nu))
    else:
        # kd_tree = KDTree(V, leaf_size=leaf_size)
        # dist, ind = kd_tree.query(V, k=k + 1)

        # adj = []
        # for i in range(len(V)):
        #     nu = [(ind[i, j], dist[i, j]) for j in range(1, len(dist[i]))]
        #     adj.append((i, nu))

        kd_tree = KDTree(V, leaf_size=leaf_size)
        dist, ind = kd_tree.query(V, k=k + 1)

        print(type(dist), type(dist[0]), dist.shape)

        temp = {}
        for i in range(len(V)):
            temp[i] = []
        for i in range(len(V)):
            for j in range(1,len(dist[i])):
                if (ind[i, j], dist[i, j]) not in temp[i]:
                    temp[i].append((ind[i, j], dist[i, j]))
                if (i, dist[i, j]) not in temp[ind[i,j]]:
                    temp[ind[i,j]].append((i, dist[i, j]))    

        adj = []
        for i in range(len(V)):
            nu = temp[i]
            adj.append((i, nu))

    return adj


# num buckets = log_(1 + beta) (W)
def create_buckets(E, alpha, beta, W):
    num_buckets = math.ceil(math.log(W, (1 + beta)))
    buckets = []
    prev_end = 0
    for i in range(num_buckets):
        now_end = np.power((1 + beta), i) + (np.random.uniform(-alpha, alpha) * np.power((1 + beta), i))
        if i < num_buckets - 1:
            buckets.append((prev_end, now_end))
            prev_end = now_end
        else:
            buckets.append((prev_end, W + 0.00001))

    bucket_counter = [0] * len(buckets)

    for key in E:
        for edge in E[key]:
            bucket_number = 1
            for bucket in buckets:
                if bucket[0] <= E[key][edge] < bucket[1]:
                    E[key][edge] = bucket_number
                    bucket_counter[bucket_number - 1] += 1
                    break
                bucket_number += 1
    return E, buckets, bucket_counter


def shift_edge_weights(E, gamma=0.05):
    max_weight = 0
    for key in E:
        for edge in E[key]:
            # TODO: fix shift (remove 100 *)
            if key < edge:
                E[key][edge] = 100 * max(E[key][edge] + (np.random.uniform(-gamma, gamma)) * E[key][edge], 0)
                max_weight = max(E[key][edge], max_weight)
            else:
                E[key][edge] = E[edge][key]
    return E, max_weight


def find_differences(contracted_leader_list):
    diff_matrix = []
    for cl in contracted_leader_list:
        diff = []
        for cl2 in contracted_leader_list:
            diff_count = 0
            for i in range(len(cl2)):
                if cl[i] != cl2[i]:
                    diff_count += 1
            diff.append(diff_count)
        diff_matrix.append(diff)
    return diff_matrix



def main():
    parser = ArgumentParser()
    parser.add_argument('--epsilon', help='epsilon [default=1/8]', type=float, default=1 / 8)
    parser.add_argument('--machines', help='Number of machines [default=1]', type=int, default=1)
    parser.add_argument('--buckets', help='Use buckets [default=False]', action='store_true')
    parser.add_argument('--getdata', help='save data to file', action='store_true')
    parser.add_argument('--datasets', help='use sklearn datasets', action='store_true')
    parser.add_argument('--test', help='Test', action='store_true')
    args = parser.parse_args()

    print('Start generating MST')
    if args.test:
        print('Test argument given')

    start_time = datetime.now()
    print('Starting time:', start_time)

    file_location = 'Results_buckets/'
    plotter = Plotter(None, None, file_location)

    # Read data
    data_reader = DataReader()

    # Parameters
    beta = 0.2  # 0 <= beta <= 1 (buckets)
    alpha = 0.1  # shift of buckets
    gamma = 0.05  # shift of edge weights


    # DO BINARY SEARCH
    L = 1.0
    R = 4.0
    mid = (L+R)/2
    RANGE = R-L
    while RANGE>=0.015 and L<=R:
        r2 = mid
        print(f'checking r2={r2}')
        dataset = get_clustering_data(r2)
        V = [item for item in dataset[0][0]]
        runs, graph, yhats, contracted_leader, mst, leaders = run(V, 8, data_reader, beta, alpha, gamma,
                                                        buckets=args.buckets, plotter=plotter)
        if len(graph) < 2:
            R = mid
        else:
            L = mid
        mid = (L+R)/2
        RANGE = R-L

    print(f'final ans: r2={L},{R}')



def run(V, k, data_reader, beta=0.0, alpha=0.0, gamma=0.0, buckets=False, plotter=None):
    if buckets:
        E, size, vertex_coordinates, W = data_reader.create_distance_matrix(V, full_dm=True)
        E, W = shift_edge_weights(E, gamma)
        adjacency_list = get_nearest_neighbours(E, k, buckets=True)
    else:
        adjacency_list = get_nearest_neighbours(V, k)
    # return affinity_clustering(adjacency_list, vertex_coordinates=None, plot_intermediate=False)
    return affinity_clustering(adjacency_list, vertex_coordinates=V, plot_intermediate=False, plotter=plotter)


# determine number of connections for each point based on local density
def degree_by_local_density(V):
    degree = [0] * len(V)


    return degree

if __name__ == '__main__':
    # Initial call to main function
    main()

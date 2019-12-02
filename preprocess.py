import networkx as nx
import numpy as np
from student_utils import *
import utils

def preprocess(num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix, scale):
    """
    Preprocess adjacency matrix

    Output:
    G: networkx graph,
    map_locations: mapping from locations to index { String -> Integer },
    product_prices: { location_index -> {house_index -> price} },
    shortest_paths: shortest paths found from the original adjacency matrix, Dictionary: { node1 -> { node2 -> [shortest_path] } }
    """
    # a. Scale all weights by scale
    adjacency_matrix = [[int(w*scale) if w != 'x' else w for w in row] for row in adjacency_matrix]

    # b. Generate a mapping from locations to node index in a hashmap
    map_locations = {}
    index = 0
    for location in list_locations:
        map_locations[location] = index
        index += 1

    # c. Put adjacency matrix in networkx
    G = nx.Graph()
    for i in range(num_of_locations):
        G.add_node(i)
    for j in range(len(adjacency_matrix)):
        for k in range(j, len(adjacency_matrix[0])):
            if (j == k):
                G.add_edge(j, k)
                G.edges[j, k]['weight'] = 0
            else:
                weight = adjacency_matrix[j][k]
                if weight != 'x':
                    G.add_edge(j, k)
                    G.edges[j, k]['weight'] = weight

    # d. Run networkx all_pairs_shortest_path
    shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    # e. Run floyd-warshall to get all pair-wise shortest distances 
    shortest_paths_len = nx.floyd_warshall(G)

    # f. Complete the graph using pair-wise shortest distances
    for m in range(num_of_locations):
        for n in range(m, num_of_locations):
            if not G.has_edge(m, n):
                G.add_edge(m, n)
            G.edges[m, n]['weight'] = shortest_paths_len[m][n]
    # TODO Noticement! Prune and Modify Edge Weights to Satisfy TSP solver
    #0.save a copy of G for other purpose
    #TODO
    #1.find max_pairwise_distance
    max_dis = G.edges[0, 0]['weight']
    for m in range(num_of_locations):
        for n in range(m, num_of_locations):
            if G.edges[m, n]['weight'] > max_dis:
                max_dis = G.edges[m, n]['weight']
    #2. rescale edges based on max_dis
    upper_bound = 2**29 - 1
    for m in range(num_of_locations):
        for n in range(m, num_of_locations):
            G.edges[m, n]['weight'] =  int((G.edges[m, n]['weight'] / max_dis) * upper_bound)
    # g. Generate product prices using pair-wise shortest distances
    product_prices = {}
    for m in range(num_of_locations):
        local_prices = {}
        for n in range(num_houses):
            index_house = map_locations[list_houses[n]]
            local_prices[index_house] = G.edges[m, index_house]['weight']
        product_prices[m] = local_prices

    import math
    # h. Update all edges in G to 2/3 & round to int
    for m in range(num_of_locations):
        for n in range(m + 1, num_of_locations):
            G.edges[m, n]['weight'] = int(G[m][n]['weight'] * 2/3)

    return G, map_locations, product_prices, shortest_paths

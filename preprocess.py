import networkx as nx
import numpy as np
from student_utils import *
import utils

def preprocess(num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix):
    """
    Preprocess adjacency matrix
    b. Generate a mapping from locations to node index in a hashmap
    c. Put adjacency matrix in networkx
    d. Run networkx all_pairs_shortest_path
    e. Run floyd-warshall to get all pair-wise shortest distances
    f. Complete the graph using pair-wise shortest distances
    g. Generate product prices using pair-wise shortest distances
    h. Update all edges in G to 2/3

    return the mapping from locations to index, the product prices(dict({location_index, dict({house_index, price})})),
    and the networkx preprocessed graph
    """

    # a. Scale all weights by SCALE
    SCALE = 10000000
    adjacency_matrix = [[int(w*SCALE) if w != 'x' else w for w in row] for row in adjacency_matrix]
    print(adjacency_matrix)

    #part b
    map_locations = {}
    index = 0
    for location in list_locations:
        map_locations[location] = index
        index += 1

    #part c
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

    #part d, e
    length = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    #part f
    for m in range(num_of_locations):
        for n in range(m, num_of_locations):
            if not G.has_edge(m, n):
                G.add_edge(m, n)
            G.edges[m, n]['weight'] = length[m][n]

    #part g
    product_prices = {}
    for m in range(num_of_locations):
        local_prices = {}
        for n in range(num_houses):
            index_house = map_locations[list_houses[n]]
            local_prices[index_house] = G.edges[m, index_house]['weight']
        product_prices[m] = local_prices

    #part h
    for m in range(num_of_locations):
        for n in range(m + 1, num_of_locations):
            G.edges[m, n]['weight'] *= 2/3

    return map_locations, product_prices, G

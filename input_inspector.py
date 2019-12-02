import networkx as nx
from glob import iglob
import utils
from student_utils import data_parser
import itertools

for i in iglob('inputs/*'):
    input_data = utils.read_file(str(i))
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    # Put adjacency matrix in networkx
    G = nx.Graph()
    for k in range(num_of_locations):
        G.add_node(k)
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

    shortest_paths_len = nx.floyd_warshall(G)
    lengths = set()
    for v in shortest_paths_len:
        for u in shortest_paths_len[v]:
            lengths.add(shortest_paths_len[v][u])
    adjacency_matrix = list(filter(lambda x: x != 'x', itertools.chain.from_iterable(adjacency_matrix)))
    print(i, 'max', max(lengths), 'min', min(adjacency_matrix))


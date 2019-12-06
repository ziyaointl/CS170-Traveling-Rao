import re
import sys
sys.path.append('..')
sys.path.append('../..')
import time
import os
import argparse
import copy as cp
import networkx as nx
import numpy as np
import io
from concorde.tsp import TSPSolver
from collections import defaultdict
from glob import iglob

def get_files_with_extension(directory, extension):
    files = []
    for name in os.listdir(directory):
        if name.endswith(extension):
            files.append('{}/{}'.format(directory, name))
    return files


def read_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [line.replace("Â", " ").strip().split() for line in data]
    return data


def write_to_file(file, string, append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        f.write(string)


def write_data_to_file(file, data, separator, append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        for item in data:
            f.write('{}/{}'.format(item, separator))


def input_to_output(input_file, output_directory):
    return (
        os.path.join(output_directory, os.path.basename(input_file))
        .replace("input", "output")
        .replace(".in", ".out")
    )

def decimal_digits_check(number):
    number = str(number)
    parts = number.split('.')
    if len(parts) == 1:
        return True
    else:
        return len(parts[1]) <= 5


def data_parser(input_data):
    number_of_locations = int(input_data[0][0])
    number_of_houses = int(input_data[1][0])
    list_of_locations = input_data[2]
    list_of_houses = input_data[3]
    starting_location = input_data[4][0]

    adjacency_matrix = [[entry if entry == 'x' else float(entry) for entry in row] for row in input_data[5:]]
    return number_of_locations, number_of_houses, list_of_locations, list_of_houses, starting_location, adjacency_matrix


def adjacency_matrix_to_graph(adjacency_matrix):
    node_weights = [adjacency_matrix[i][i] for i in range(len(adjacency_matrix))]
    adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in adjacency_matrix]

    for i in range(len(adjacency_matrix_formatted)):
        adjacency_matrix_formatted[i][i] = 0

    G = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix_formatted))

    message = ''

    for node, datadict in G.nodes.items():
        if node_weights[node] != 'x':
            message += 'The location {} has a road to itself. This is not allowed.\n'.format(node)
        datadict['weight'] = node_weights[node]

    return G, message


def is_metric(G):
    shortest = dict(nx.floyd_warshall(G))
    for u, v, datadict in G.edges(data=True):
        if abs(shortest[u][v] - datadict['weight']) >= 0.00001:
            return False
    return True


def adjacency_matrix_to_edge_list(adjacency_matrix):
    edge_list = []
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[0])):
            if adjacency_matrix[i][j] == 1:
                edge_list.append((i, j))
    return edge_list


def is_valid_walk(G, closed_walk):
    if len(closed_walk) == 2:
        return closed_walk[0] == closed_walk[1]
    return all([(closed_walk[i], closed_walk[i+1]) in G.edges for i in range(len(closed_walk) - 1)])


def get_edges_from_path(path):
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]

"""
G is the adjacency matrix.
car_cycle is the cycle of the car in terms of indices.
dropoff_mapping is a dictionary of dropoff location to list of TAs that got off at said droppoff location
in terms of indices.
"""
def cost_of_solution(G, car_cycle, dropoff_mapping):
    cost = 0
    message = ''
    dropoffs = dropoff_mapping.keys()
    if not is_valid_walk(G, car_cycle):
        message += 'This is not a valid walk for the given graph.\n'
        cost = 'infinite'

    if not car_cycle[0] == car_cycle[-1]:
        message += 'The start and end vertices are not the same.\n'
        cost = 'infinite'
    if cost != 'infinite':
        if len(car_cycle) == 1:
            car_cycle = []
        else:
            car_cycle = get_edges_from_path(car_cycle[:-1]) + [(car_cycle[-2], car_cycle[-1])]
        if len(car_cycle) != 1:
            driving_cost = sum([G.edges[e]['weight'] for e in car_cycle]) * 2 / 3
        else:
            driving_cost = 0
        walking_cost = 0
        shortest = dict(nx.floyd_warshall(G))

        for drop_location in dropoffs:
            for house in dropoff_mapping[drop_location]:
                walking_cost += shortest[drop_location][house]

def preprocess(num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix, ratio = 0):
    """
    Preprocess adjacency matrix

    Output:
    G: networkx graph,
    map_locations: mapping from locations to index { String -> Integer },
    product_prices: { location_index -> {house_index -> price} },
    shortest_paths: shortest paths found from the original adjacency matrix, Dictionary: { node1 -> { node2 -> [shortest_path] } }
    """

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
    if ratio <= 0:
        upper_bound = 2 ** 25 - 1
        ratio = 10 ** 5
        if ratio > (upper_bound / max_dis):
            ratio = upper_bound / max_dis
    for m in range(num_of_locations):
        for n in range(m, num_of_locations):
            G.edges[m, n]['weight'] =  int(G.edges[m, n]['weight'] * ratio)
    # g. Generate product prices using pair-wise shortest distances
    product_prices = {}
    for m in range(num_of_locations):
        local_prices = {}
        for n in range(num_houses):
            index_house = map_locations[list_houses[n]]
            local_prices[index_house] = G.edges[m, index_house]['weight']
        product_prices[m] = local_prices

    # h. Update all edges in G to 2/3 & round to int
    for m in range(num_of_locations):
        for n in range(m + 1, num_of_locations):
            G.edges[m, n]['weight'] = int(G[m][n]['weight'] * 2/3)

    return G, map_locations, product_prices, shortest_paths


def superprint(*arg):
    print('------------------------------')
    print(*arg)
    print('------------------------------')

def print_graph(G):
    print(' '.join([str(e) for e in G.edges()]))

def TSP(G, nodes, start, name):
    """Returns an approximate TSP solution
    Input:
    G: networkx graph
    nodes: we want to find a cycle among these nodes
    Function might be useful TODO subgraph = G.subgraph(nodes)

    Output:
    cycle: solution
    """
    superprint('Calling TSP on', nodes, start)
    def cal_TSP_cost(G, start, cycle):
        l = len(cycle)
        assert l <= 5, "should never call naive on too complex cases!"
        travel_cost = 0
        for i in range(1, l):
            travel_cost += G[cycle[i - 1]][cycle[i]]['weight']
        return travel_cost

    def naive_TSP(G, nodes, start, name):
        l = len(nodes)
        nodes_cp = nodes[:]
        nodes_cp.remove(start)
        ans = []
        if l == 2 or l == 3:
            ans = [start] + nodes_cp + [start]
        elif l == 4:
            a,b,c = nodes_cp[0], nodes_cp[1], nodes_cp[2]
            sol1 = [start] + nodes_cp + [start]
            sol2 = [start, a, c, b, start]
            sol3 = [start, b, a, c, start]
            sol4 = [start, b ,c ,a, start]
            sol5 = [start, c, a, b, start]
            sol6 = [start, c, b, a, start]
            #3 of them are over calculated
            ans = min([sol1,sol2,sol3,sol4,sol5,sol6], key= lambda x:cal_TSP_cost(G,start,x))
            #return Google_inefficient_TSP(G,nodes,start,name)
        else:
            print("incorrect parameter for naive TSP")
        superprint('TSP call finished', ans)
        return ans

    if len(nodes) <= 4 :
        print("Calling naive TSP (made on brute force by Rui Chen)")
        if len(nodes) < 2:
            print("Too Simple testcase even for Naive TSP !")
        return naive_TSP(G, nodes, start, name)
    def gen_output(G, filename):
        """Input:
        G: graph
        filename: tsp file to be written
        """
        fout = open(filename, 'w')
        fout.write('NAME : ' + filename + '\n')
        fout.write('TYPE : TSP\n')
        fout.write('DIMENSION : ' + str(len(G.nodes())) + '\n')
        fout.write('EDGE_WEIGHT_TYPE : EXPLICIT\n')
        fout.write('EDGE_WEIGHT_FORMAT : FULL_MATRIX\n')

        fout.write('EDGE_WEIGHT_SECTION :\n')
        sorted_nodes = sorted(G.nodes())
        lines = []
        for v in sorted_nodes:
            line = []
            for w in sorted_nodes:
                line.append(str(G[v][w]['weight']))
            lines.append(' '.join(line))
        fout.write('\n'.join(lines))
        fout.write('\nEOF\n')
        fout.close()

    # Reduce G to specified node list
    G = nx.subgraph(G, nodes)

    # Generate mapping
    # Maps sorted(G.nodes()).index(concorde_index) -> node
    sorted_nodes = sorted(G.nodes())

    def concorde_index_to_node(concorde_index):
        return sorted_nodes[concorde_index]

    def node_to_concorde_index(node):
        return sorted_nodes.index(node)

    # Create data
    filename = name + ".tsp"
    gen_output(G, filename)

    # Call Concorde
    solver = TSPSolver.from_tspfile(filename)
    solution = solver.solve()

    # Transform points back & return path
    path = [concorde_index_to_node(concorde_index) for concorde_index in solution.tour]

    # Roatate path so that start is at the start
    for i in range(len(path)):
        if path[i] == start:
            path = path[i:] + path[:i] + [start]
    superprint('TSP call finished', path)
    return path

def get_input_path(filename):
    return 'inputs/' + filename + '.in'

def get_output_path(filename):
    return 'outputs/' + filename + '.out'

def main(input_data, filename):
    """Given a filename, genereate a solution, and save it to filename.out
    """
    # 1. Read input
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(
        input_data)

    # 2. Preprocess adjacency matrix
    r = 0
    if filename in ['214_200', '71_200', '17_200', '89_100', '211_200', '234_200', '72_200', '226_200', '124_100', '297_200', '254_200','100_200', '180_200','131_100', '162_200', '237_100', '12_200', '254_100', '114_200']:
        r = 1
    elif filename == ['108_200', '162_200']:
        r = 0.1
    elif filename in ['145_200', '277_200']:
        r = 0.01
    elif filename == '131_200':
        r = 100
    elif filename == '272_200':
        r = 0.001
    elif filename in ['30_200', '83_200', '228_100', '227_100']:
        r = 10000
    elif filename in ['285_100', '216_200']:
        r = 1000
    G, location_mapping, offers, shortest_paths = preprocess(num_of_locations, num_houses, list_locations, list_houses,
                                             starting_car_location, adjacency_matrix, r)
    print('locations', location_mapping)
    print('offers', offers)
    print('shortest_paths', shortest_paths)
    print(nx.to_numpy_matrix(G))
    # print('TSP', TSP(G, G.nodes(), location_mapping[starting_car_location]))

    # 3. Solve
    res = solve(G, offers, location_mapping[starting_car_location], [location_mapping[h] for h in list_houses], filename)

    # 4. Write to file
    fout = io.StringIO()
    # Reconstruct the path
    path = res['path']
    new_path = [path[0]]
    for i in range(1, len(path)):
        new_path[-1:] = shortest_paths[path[i-1]][path[i]]
    res['path'] = new_path
    # Write to file
    for location in res['path']:
        fout.write(list_locations[location] + ' ')
    dropoffs = get_dropoffs(G, res, [location_mapping[h] for h in list_houses], offers)
    dropoffs = [[list_locations[location] for location in lst] for lst in dropoffs]
    fout.write('\n' + str(len(dropoffs)) + '\n')
    fout.write('\n'.join([' '.join(lst) for lst in dropoffs]) + '\n')
    superprint('Final answer:', 'Cost:', res['cost'], '\n', 'Path:', res['path'])
    return fout.getvalue()

def get_dropoffs(G, sol, homes, offers):
    """Input:
    G: graph
    sol: dictionary, as defined in solve()
    homes: list of homes
    offers: dictionary {location -> {home -> price}}

    Output:
    res: list of locations to be printed
    """
    prices = defaultdict(lambda: float('inf')) # maps home to the cheapest dropoff price
    dropoffs = {} # maps home to the closest dropoff locations
    res = defaultdict(list)
    for location in sol['path']:
        for home in homes:
            if prices[home] > offers[location][home]:
                dropoffs[home] = location
                prices[home] = offers[location][home]
    for home in dropoffs:
        res[dropoffs[home]].append(home)
    return [[location] + res[location] for location in res]

def deepcopy(sol):
    assert type(sol) == dict, "deepcopy only works for a solution (type dictionary)!"
    return cp.deepcopy(sol)


def l_drop(G, potential_sol, l, start_location, homes, offers):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()
    l: length of path to be dropped
    start_location: start index of the path being dropped
    homes: A list of homes
    offers: Products offered at each market, Dictionary : {location -> {home -> price}}

    Output:
    new_sol: Solution after dropping l
    """
    new_sol = deepcopy(potential_sol)
    new_sol['path'] = new_sol['path'][:start_location] + new_sol['path'][start_location+l:]
    new_sol['cost'] = calc_cost(G, new_sol['path'], homes, offers)
    return new_sol

def l_consecutive_drop(G, potential_sol, l, homes, offers, start):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()
    l: start value for l. 1 <= l < len(potential_sol['path'])
    homes: A list of homes
    offers: Products offered at each market, Dictionary : {location -> {home -> price}}
    start: start node, Integer

    Output:
    potential_sol: possibly improved solution
    """
    final_sol = potential_sol
    while l > 0:
        path = potential_sol['path']
        # Find the largest cost improvement drop
        for i in range(1, len(path) - l):
            new_sol = l_drop(G, potential_sol, l, i, homes, offers)
            if (final_sol['cost'] > new_sol['cost']):
                final_sol = new_sol
        # If cost did not decrease after drop, decrease l
        if final_sol['cost'] == potential_sol['cost']:
            l -= 1
        # Else, try again with the same l
        else:
            potential_sol = final_sol
    return final_sol


def helper_add(G, sol, node, homes, offers, start, name):
    cycle = sol['path']  # cycle is a list of integer
    assert node not in cycle, "node added must not in the cycle"
    new_list = list(set(cycle + [node]))
    tsp_path = TSP(G, new_list, start, name)
    # assert len(cycle) >= 2, "current implementation cannot support minor edge case"
    """cost = sol['cost']
    curr_min = -G[cycle[0]][cycle[1]]['weight'] + G[cycle[0]][node]['weight'] + G[cycle[1]][node]['weight']
    curr_cycle = cycle[:].insert(1, node)
    for node_i in cycle:
        for node_j in cycle:
            if node_i == node_j:
                continue
            else:
                s = -G[node_i][node_j]['weight'] + G[node_i][node]['weight'] + G[node_j][node]['weight']"""
    new_sol = {}
    new_sol['path'] = tsp_path
    new_sol['cost'] = calc_cost(G, tsp_path, homes, offers)
    return new_sol


def insert(G, potential_sol, homes, offers, start, name):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()
    homes: list of homes
    offers: list of offers
    start: start node

    Output:
    potential_sol: possibly improved solution
    """
    # Add a new vertex to the current solution if such insertion implies a reduction in total cost
    # the node that must maximize the reduction in cost!
    optimal = potential_sol
    total = list(G.nodes)
    li_dif = [i for i in total if i not in potential_sol['path']]
    for node in li_dif:
        if node in potential_sol['path']:
            print("edge_cases responsible by Rui Chen in insert")
            continue
        temp = helper_add(G, potential_sol, node, homes, offers, start, name)
        if temp['cost'] < optimal['cost']:
            optimal = temp
    return optimal


def shake(G, potential_sol, phi, homes, offers, start, name):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()
    phi: additional percent cost permitted in new sol

    Output:
    potential_sol: perturbed solution
    """
    potential_sol_final = deepcopy(potential_sol)
    total = list(G.nodes)
    li_dif = [i for i in total if i not in potential_sol['path']]
    for node in li_dif:
        potential_sol_cp = deepcopy(potential_sol)

        #    add_node adds in the order that minimizes x_1 + x_2 - y
        #    (x_1,x_2 represents the distance to the node, y represents the distance between two vertexs in the existing circle)

        potential_sol_cp = helper_add(G, potential_sol_cp, node, homes, offers, start, name)
        if potential_sol_cp['cost'] < (phi + 1) * potential_sol['cost']:
            potential_sol_final = helper_add(G, potential_sol_final, node, homes, offers, start, name)
    return potential_sol_final


def verify_path(G, path, homes, start):
    assert len(path) > 0, "Nothing in path"
    assert path[0] == start and path[-1] == start, "First and last nodes have to be start"


def calc_cost(G, path, homes, offers):
    """Input:
    G: a complete graph
    path: valid candidate path, List : [Integer]
    homes: A list of homes
    offers: Products offered at each market, Dictionary : {location -> {home -> price}}

    Output:
    res: a single integer, cost of the current path
    """
    cheapest_prices = defaultdict(lambda: float('inf'))
    travel_cost = 0
    for l in path:
        for h in homes:
            cheapest_prices[h] = min(cheapest_prices[h], offers[l][h])
    for i in range(1, len(path)):
        travel_cost += G[path[i - 1]][path[i]]['weight']
    return sum([cheapest_prices[h] for h in homes]) + travel_cost


def solve(G, offers, start, homes, name, l=10, phi=0.35, phi_delta=0.01):
    """Input:
    G: Complete graph
    offers: Products offered at each market, Dictionary : {location -> {home -> price}}
    start: Start location, Integer
    homes: Node names of homes, List : [Integer]
    l: Longest path (number of nodes) to remove, Integer
    phi: Percent threshold for shake(), Float
    phi_delta: Percent decrease of phi after each iteration, Float

    Output:
    sol: Dictionary
    {'path' : A feasible cycle that may contain pseudo-edges List : [Integer]
     'cost' : Cost for the solution}
    """
    # A dictionary of solution info
    sol = {
        'path': TSP(G, G.nodes(), start, name)
    }
    sol['cost'] = calc_cost(G, sol['path'],
                            homes, offers)
    potential_sol = deepcopy(sol)
    while phi > 0:
        while True:
            potential_sol = l_consecutive_drop(G, potential_sol, l, homes, offers, start)
            print('After dropping', potential_sol)
            verify_path(G, potential_sol['path'], homes, start)
            potential_sol = insert(G, potential_sol, homes, offers, start, name)
            verify_path(G, potential_sol['path'], homes, start)
            if potential_sol['cost'] < sol['cost']:
                sol = potential_sol
            else:
                break
        potential_sol = shake(G, potential_sol, phi, homes, offers, start, name)
        verify_path(G, potential_sol['path'], homes, start)
        phi -= phi_delta
    return sol

def get_all_inputs():
    res = list(iglob('inputs/*_200.in')) + list(iglob('inputs/*_50.in')) + list(iglob('inputs/*_100.in'))
    return [i.split('/')[1].split('.')[0] for i in res]

def get_all_outputs():
    res = list(iglob('outputs/*'))
    return set([i.split('/')[1].split('.')[0] for i in res])

def get_all_failures():
    f = open('log.txt', 'r')
    res = set()
    for l in f:
        m = re.search(r'failed', l)
        if m:
            res.add(m.string.strip().split(' ')[0])
    return res

if __name__ == '__main__':
    # TODO: Google TSP and spindly graph test
    if len(sys.argv) > 1:
        name = sys.argv[1]
        res = main(read_file(get_input_path(name)), name)
        fout = open(get_output_path(name), 'w')
        fout.write(res)
        print('Wrote', name)
        fout.close()
        exit(0)
    from dask.distributed import Client
    from tornado.util import TimeoutError
    import traceback
    all_inputs = get_all_inputs()
    existing_outputs = get_all_outputs()
    failures = get_all_failures()
    client = Client("tcp://34.83.248.90:8786")
    tasks = list(filter(lambda x: (x not in existing_outputs) and (x not in failures), all_inputs))
    print('failures', failures)
    print('tasks', tasks)
    done_tasks = set()
    futures = []
    for t in tasks:
        future = client.submit(main, read_file(get_input_path(t)), t)
        futures.append(future)
    while len(done_tasks) < len(futures):
        for i in range(len(futures)):
            f = futures[i]
            t = tasks[i]
            if t not in done_tasks and f.done():
                done_tasks.add(t)
                try:
                    res = f.result(1)
                    fout = open(get_output_path(t), 'w')
                    fout.write(res)
                    print('Wrote', t)
                    fout.close()
                except TimeoutError as e:
                    print(e)
                    print('Could not gather result {}, retrying...'.format(t))
                    print(traceback.format_exc())
                    done_tasks.remove(t)
                except Exception:
                    print(t, 'failed')
                    print(traceback.format_exc())
        time.sleep(10)
        print('Tick')

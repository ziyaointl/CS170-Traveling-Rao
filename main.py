import sys
sys.path.append('..')
sys.path.append('../..')
import os
import argparse
import utils
import copy as cp
import networkx as nx
import numpy as np
from student_utils import *
from concorde.tsp import TSPSolver
from collections import defaultdict
from preprocess import preprocess

def superprint(*arg):
    print('------------------------------')
    print(*arg)
    print('------------------------------')

def print_graph(G):
    print(' '.join([str(e) for e in G.edges()]))

def TSP(G, nodes, start):
    """Returns an approximate TSP solution
    Input:
    G: networkx graph
    nodes: we want to find a cycle among these nodes
    Function might be useful TODO subgraph = G.subgraph(nodes)

    Output:
    cycle: solution
    """
    superprint('Calling TSP on', nodes, start)
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
    filename = "test.tsp"
    gen_output(G, "test.tsp")

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

def main(filename='50'):
    """Given a filename, genereate a solution, and save it to filename.out
    """
    # 1. Read input
    input_data = utils.read_file('inputs/' + str(filename) + '.in')
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(
        input_data)
    SCALE = 100000

    # 2. Preprocess adjacency matrix
    G, location_mapping, offers, shortest_paths = preprocess(num_of_locations, num_houses, list_locations, list_houses,
                                             starting_car_location, adjacency_matrix, SCALE)
    print('locations', location_mapping)
    print('offers', offers)
    print('shortest_paths', shortest_paths)
    print(nx.to_numpy_matrix(G))  
    # print('TSP', TSP(G, G.nodes(), location_mapping[starting_car_location]))
    
    # 3. Solve
    res = solve(G, offers, location_mapping[starting_car_location], [location_mapping[h] for h in list_houses])

    # 4. Write to file
    fout = open(get_output_path(filename), 'w')
    # Reconstruct the path
    path = res['path']
    new_path = [path[0]]
    for i in range(1, len(path)):
        print(new_path)
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
    fout.close()

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


def helper_add(G, sol, node, homes, offers, start):
    cycle = sol['path']  # cycle is a list of integer
    assert node not in cycle, "node added must not in the cycle"
    new_list = cycle[:] + [node]
    tsp_path = TSP(G, new_list, start)
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


def insert(G, potential_sol, homes, offers, start):
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
        temp = helper_add(G, potential_sol, node, homes, offers, start)
        if temp['cost'] < optimal['cost']:
            optimal = temp
    return optimal


def shake(G, potential_sol, phi, homes, offers, start):
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

        potential_sol_cp = helper_add(G, potential_sol_cp, node, homes, offers, start)
        if potential_sol_cp['cost'] < (phi + 1) * potential_sol['cost']:
            potential_sol_final = helper_add(G, potential_sol_final, node, homes, offers, start)
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


def solve(G, offers, start, homes, l=10, phi=0.35, phi_delta=0.01):
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
        'path': TSP(G, G.nodes(), start)
    }
    sol['cost'] = calc_cost(G, sol['path'],
                            homes, offers)
    potential_sol = deepcopy(sol)  # TODO: Implement deepcopy
    while phi > 0:
        while True:
            potential_sol = l_consecutive_drop(G, potential_sol, l, homes, offers, start) 
            verify_path(G, potential_sol['path'], homes, start)
            potential_sol = insert(G, potential_sol, homes, offers, start)
            verify_path(G, potential_sol['path'], homes, start)
            if potential_sol['cost'] < sol['cost']:
                sol = potential_sol
            else:
                break
        potential_sol = shake(G, potential_sol, phi, homes, offers, start)
        verify_path(G, potential_sol['path'], homes, start)
        phi -= phi_delta
    return sol

if __name__ == '__main__':
    main('50')

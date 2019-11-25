import sys
sys.path.append('..')
sys.path.append('../..')
import os
import argparse
import utils
import networkx as nx
import numpy as np
from student_utils import *
from concorde.tsp import TSPSolver
from collections import defaultdict


def main(filename='50'):
    """Given a filename, genereate a solution, and save it to filename.out
    """
    # 1. Read input
    input_data = utils.read_file('inputs/' + str(filename) + '.in')
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)

    # 2. Preprocess adjacency matrix
    # a. Generate a mapping from locations to node index in a hashmap
    # b. Put adjacency matrix in networkx
    # c. Remove all node not connected to start
    # d. Run networkx all_pairs_shortest_path
    # e. Run floyd-warshall to get all pair-wise shortest distances
    # f. Complete the graph using pair-wise shortest distances
    # g. Generate product prices using pair-wise shortest distances
    # h. Update all edges in G to 2/3
    location_mapping, offers, G = preprocess(num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix)

    # 3. Solve
    res = solve(G, offers, start_car_location)

    # 4. Write to file

def deepcopy(sol):
    assert type(sol) == dict, "deepcopy only works for dictionary!"
    pass

def l_consecutive_drop(G, potential_sol, l):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()
    l: start value for l. 1 <= l < len(potential_sol['path'])

    Output:
    potential_sol: possibly improved solution
    """
    while l > 0:
        for i in range(l):
            pass
    pass

def helper_add(G, sol, node):
    cycle = sol['path'] #cycle is a list of integer
    assert node not in cycle, "node added must not in the cycle"
    new_list = cycle[:] + [node]
    solution = TSP(G, new_list)
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
    new_sol['path'] = solution
    new_sol['cost'] = calc_cost(G, solution, homes, offers)
    return new_sol

def insert(G, potential_sol):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()

    Output:
    potential_sol: possibly imporved solution
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
        temp = helper_add(G, potential_sol, node)
        if temp['cost'] < optimal['cost']:
            optimal = temp
    return optimal



def shake(G, potential_sol, phi, phi_delta):
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

       potential_sol_cp = helper_add(G, potential_sol_cp, node)
       if potential_sol_cp['cost'] < (phi + 1) * potential_sol['cost']:
            potential_sol_final = helper_add(G, potential_sol_final, node)
    return potential_sol_final


def verify_path(G, path, homes, start):
    assert len(path) > 0, "Nothing in path"
    assert path[0] == start and path[-1] == start, "First and last nodes have to start"

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
    for l in path:
        for h in homes:
            cheapest_prices[h] = min(cheapest_prices[h], offers[l][h])
    return sum([cheapest_prices[h] for h in homes])

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
            'path': TSP(G, s)
          }
    sol['cost'] = calc_cost(G, sol['path'], homes) # TODO: Write calc_cost, with reference to a similar function in student_utils.py
    potential_sol = deepcopy(sol) # TODO: Implement deepcopy
    while phi > 0:
        while True:
            potential_sol = l_consecutive_drop(potential_sol, l)
            potential_sol = insert(G, potential_sol)
            if potential_sol['cost'] < sol['cost']:
                sol = potential_sol
            else:
                break
        shake(G, potential_sol, phi, phi_delta)
        phi -= phi_delta
    return sol

def TSP(G, nodes):
    """Returns an approximate TSP solution
    Input:
    G: networkx graph
    nodes: we want to find a cycle among these nodes
    Function might be useful TODO subgraph = G.subgraph(nodes)

    Output:
    cycle: solution
    """
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
    	fout.write('EOF\n')
    	fout.close()

    # Reduce G to specified node list
    G = nx.subgraph(nodes)

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
    return [concorde_index_to_node(concorde_index) for concorde_index in solution.tour]

if __name__ == '__main__':
    main()


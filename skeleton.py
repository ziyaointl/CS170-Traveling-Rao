import sys
sys.path.append('..')
sys.path.append('../..')
import os
import argparse
import utils
import networkx as nx
import numpy as np
from student_utils import *

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

    # 3. Solve
    res = solve(G, offers, start_car_location)

    # 4. Write to file

def deepcopy(sol):
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
    new_sol = deepcopy(sol)
    new_sol['path'] = solution
    new_sol['cost'] = calc_cost(G, )
    return

def insert(G, potential_sol):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()

    Output:
    potential_sol: possibly imporved solution
    """
    # Add a new vertex to the current solution if such insertion implies a reduction in total cost
    # the node that must maximize the reduction in cost!
    for node in list(G.nodes):
        pass
    pass



def shake(G, potential_sol, phi):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()
    phi: additional percent cost permitted in new sol

    Output:
    potential_sol: perturbed solution
    """
    # potential_sol_final = deepcopy(potential_sol)
    # for node in G\sol:
    #   potential_sol_cp = deep_copy(potential_sol)
    #   potential_sol_cp = add_node(potential_sol_cp, node)
    #   # TODO add_node adds in the order that minimizes x_1 + x_2 - y
    #   # (x_1,x_2 represents the distance to the node, y represents the distance between two vertexs in the existing circle)
    #   if potential_sol_cp['cost'] < phi * potential_col['cost']:
    #        potential_sol_final = add_node(potential_sol_final, node)
    # return potential_sol_final
    pass

def verify_path(G, path, homes, start):
    assert len(path) > 0, "Nothing in path"
    assert path[0] == start and path[-1] == start, "First and last nodes have to start"

def calc_cost(G, path, homes, offers):
    """Input:
    G: a complete graph
    path: valid candidate path
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
            potentail_sol = insert(potential_sol)
            if potential_sol['cost'] < sol['cost']:
                sol = potential_sol
            else:
                break
        phi -= phi_delta
        shake(potential_sol)
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
    pass

if __name__ == '__main__':
    main()


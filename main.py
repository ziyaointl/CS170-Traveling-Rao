import sys
sys.path.append('..')
sys.path.append('../..')
import time
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

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp



def create_data_model(G,start,nodes):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = nx.to_numpy_matrix(G).tolist()
    print(data['distance_matrix'])  # yapf: disable
    data['num_vehicles'] = 1
    data['depot'] = start
    return data


def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    #print('Objective: {} miles'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = []
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    #plan_output += ' {}\n'.format(manager.IndexToNode(index))
    #print(plan_output)
    #plan_output += 'Route distance: {}miles\n'.format(route_distance)
    return plan_output


def Google_inefficient_TSP(G,nodes,start,name):
    """Entry point of the program."""
    # Instantiate the data problem.
    G = nx.subgraph(G, nodes)
    data = create_data_model(G,start,nodes)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        a = print_solution(manager, routing, assignment)
        a = a + [a[0]]
        print(a)
    return a




"""if __name__ == '__main__':
    Google_inefficient_TSP()"""


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
            travel_cost += G[path[i - 1]][path[i]]['weight']
        return travel_cost

    def naive_TSP(G, nodes, start, name):
        l = len(nodes)
        nodes_cp = nodes[:]
        nodes_cp.remove(start)
        if l == 2:
            return [start] + nodes_cp + [start]
        elif l == 3:
            sol1 = [start] + nodes_cp + [start]
            sol2 = sol1[:]
            sol2.reverse()
            if cal_TSP_cost(G, start, sol1) > cal_TSP_cost(G, start, sol2):
                return sol2
            else:
                return sol1
        elif l == 4:
            return Google_inefficient_TSP(G,nodes,start,name)
        else:
            print("incorrect parameter for naive TSP")

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

def main(filename='50'):
    """Given a filename, genereate a solution, and save it to filename.out
    """
    # 1. Read input
    input_data = utils.read_file('inputs/' + str(filename) + '.in')
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(
        input_data)
    SCALE = 10000

    # 2. Preprocess adjacency matrix
    G, location_mapping, offers, shortest_paths = preprocess(num_of_locations, num_houses, list_locations, list_houses,
                                             starting_car_location, adjacency_matrix, SCALE)
    print('locations', location_mapping)
    print('offers', offers)
    print('shortest_paths', shortest_paths)
    print(nx.to_numpy_matrix(G))
    # print('TSP', TSP(G, G.nodes(), location_mapping[starting_car_location]))

    # 3. Solve
    res = solve(G, offers, location_mapping[starting_car_location], [location_mapping[h] for h in list_houses], filename)

    # 4. Write to fil
    fout = open(get_output_path(filename), 'w')
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


def helper_add(G, sol, node, homes, offers, start, name):
    cycle = sol['path']  # cycle is a list of integer
    assert node not in cycle, "node added must not in the cycle"
    new_list = cycle[:] + [node]
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

if __name__ == '__main__':
    main('1_50')
    # TODO: Automatic task discovery
    # TODO: Automatic completion detection
    # TODO: Adaptive graph weight handling
    # TODO: Error handling
    # TODO: Kubernetes
    # TODO: Google TSP and spindly graph test
    #from dask.distributed import Client, LocalCluster
    #cluster = LocalCluster()
    #client = Client(cluster)
    #tasks = [str(i) + '_50' for i in range(1, 8)]
    #futures = []
    #for t in tasks:
    #    future = client.submit(main, t)
    #    futures.append(future)
    #ans = [f.result() for f in futures]


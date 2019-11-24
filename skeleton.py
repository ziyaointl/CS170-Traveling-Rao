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
        for i in range()

def insert(G, potential_sol):
    """Input:
    G: graph
    potential_sol: dictionary, as defined in solve()

    Output:
    potential_sol: possibly imporved solution
    """
    # Add a new vertex to the current solution if such insertion implies a reduction in total cost
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

def calc_cost(G, path, homes):
    pass

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

def TSP(G, s):
    """Returns an approximate TSP solution
    Input:
    G: networkx graph
    s: start node

    Output:
    cycle: solution
    """
    pass

if __name__ == '__main__':
    main()


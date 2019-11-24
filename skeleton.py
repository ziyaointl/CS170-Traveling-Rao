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

def solve(G, offers, start, homes, l=10, phi=35):
    """Input:
    G: Complete graph
    offers: Products offered at each market, Dictionary : {location -> {home -> price}}
    start: Start location, Integer
    homes: Node names of homes, List : [Integer]
    l: Longest path (number of nodes) to remove, Integer
    phi: Percent threshold for shake(), Float

    Output:
    A feasible cycle that may contain pseudo-edges List : [Integer]
    """
    

if __name__ == '__main__':
    main()


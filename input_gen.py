import sys
sys.path.append('..')
sys.path.append('../..')
import random
import os
import argparse
import utils
import networkx as nx
import numpy as np
from student_utils import *

RANGE_OF_INPUT_SIZES = [50, 100, 200]
VALID_FILENAMES = ['50.in', '100.in', '200.in']
MAX_NAME_LENGTH = 20

def get_random_node(max_x, max_y):
    """Generate a random node
    """
    x = np.random.randint(0,max_x, None, 'int64')
    y = np.random.randint(0, max_y, None, 'int64')
    x = x/10000 # enforce 5 decimal float
    y = y/10000
    return [x,y]

def get_dis(x, y):
    return round(((x[1] - y[1])**2 + (x[2] - y[2])**2)**0.5, 5)

def get_random_array(n, p, node_list):
    """Calculate distance between each point. If rand > p, then no edge.
    Also no self edge is allowed.
    Output: Adjacency matrix
    """
    result = []
    # Preven self edges
    for i in range(n):
        temp = []
        for j in range(n):
            if (i == j):
                temp.append("x")
            else:
                temp.append(get_dis(node_list[i], node_list[j]))
        result.append(temp)
    # Stochastically add edges
    for i in range(n):
        for j in range(n):
            new_prob = np.random.random_sample()
            if (new_prob > p):
                result[i][j] = "x"
                result[j][i] = "x"
    return result

def not_all_x(lst):
    for i in lst:
        if i != "x":
            return True
    return False

#   n is one of [50, 100, 200]
# 1. Better error handling
# 2. Better filenames
def write_input_file(n, h):
    """Input
       n: locations, int
       h: homes, int

       Output:
       write to file in folder inputs/
       res: adjacency matrix
    """
    RETRIES = 100 # Number of retries before giving up
    for retry in range(RETRIES):
        total = n
        EDGE_PROB = 0.6 # default probabilty of an edge
        node_list = [] # list of (name, position), position = (x, y)
        # Generate nodes
        for i in range(total):
            # sqrt(2e14) so the longest edge won't exceed the limit
            temp = get_random_node(14142130, 14142130)
            node_list.append(["index" + str(i)] + temp)

        # Generate randomized edges
        # res: adjacency matrix to be returned
        res = get_random_array(total, EDGE_PROB, node_list)

        # Find the largest SCC
        location = [i[0] for i in node_list] # Extract locations names
        # Read matrix into an nx graph
        G = nx.from_numpy_matrix(np.matrix(res), False, nx.Graph)
        for i in range(total):
            for j in range(i, total):
                if res[i][j] == "x":
                    G.remove_edge(i, j)

        # Find the largest cc
        largest_cc = list(max(nx.connected_components(G), key=len))

        # Check if largest cc is valid
        print(largest_cc)
        if len(largest_cc) < h:
            continue

        # Choose h homes from the largest cc
        random.shuffle(largest_cc)
        home = largest_cc[0:h]
        home.sort()
        home = [ "index"+ str(i) for i in home]

        # Write to output
        f = open("inputs/{}L_{}H.in".format(n, h), "w")
        f.write(str(total) + "\n")
        l = len(home)
        f.write(str(l) + "\n")

        for i in range(total - 1):
            f.write(str(location[i]) + " ")
        f.write(str(location[total - 1]) + "\n")
        for i in range(l - 1):
            f.write(str(home[i]) + " ")
        f.write(str(home[l - 1]) + "\n")
        f.write(str(random.choice(home)) + "\n")
        for i in range(total):
            for j in range(total - 1):
                f.write(str(res[i][j]) + " ")
            f.write(str(res[i][total-1]) + "\n")
        print('Found graph in {} tries.'.format(retry + 1))
        return res
    raise RuntimeException('No valid graphs found within {} tries.'.format(RETRIES))

params = [(50, 25), (100, 50), (200, 100)]
for p in params:
    write_input_file(p[0], p[1])


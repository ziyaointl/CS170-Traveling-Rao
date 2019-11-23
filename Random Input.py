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

def get_random_array(n, p, lista):
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
                temp.append(get_dis(lista[i], lista[j]))
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
    """n: locations, int
       h: homes, int
    """
    #total = np.random.randint(0, n) + 1
    total = n
    prob = 0.6 # default probabilty of an edge
    lista = [] # list of (name, position), position = (x, y)
    # Generate nodes
    for i in range(total):
        # 1.41*10**(9 + 5) is the largest 
        temp = get_random_node(1.41*10 ** (9 + 5), 1.41*10 ** (9 + 5))
        lista.append(["index" + str(i)] + temp)
    
    # Generate randomized edges
    point_array = get_random_array(total, prob, lista)

    # Find the largest SCC
    location = [i[0] for i in lista] # Extract locations names
    Adjacency_matrix = np.matrix(point_array)
    # Read matrix into nx
    G = nx.from_numpy_matrix(Adjacency_matrix, False, nx.Graph)
    for i in range(total):
        for j in range(i, total):
            if point_array[i][j] == "x":
                G.remove_edge(i, j)

    # Find the largest cc
    largest_cc = list(max(nx.connected_components(G), key=len))

    # Check if largest cc is valid
    if len(largest_cc) < h:
        return "no enough homes"

    # Choose h homes from the largest cc
    random.shuffle(largest_cc)
    home = largest_cc[0:h]
    home.sort()
    home = [ "index"+ str(i) for i in home]

    # Write to output
    f = open("input.in", "w")
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
            f.write(str(point_array[i][j]) + " ")
        f.write(str(point_array[i][total-1]) + "\n")
    return point_array

print(write_input_file(200, 100))

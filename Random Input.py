import sys
sys.path.append('..')
sys.path.append('../..')
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
    x = np.random.randint(0,max_x, None, 'int64')
    y = np.random.randint(0, max_y, None, 'int64')
    x = x/10000
    y = y/10000
    return [x,y]

def get_dis(x, y):
    return round(((x[1] - y[1])**2 + (x[2] - y[2])**2)**0.5, 5)

def get_random_array(n, p, lista):
    result = []
    for i in range(n):
        temp = []
        for j in range(n):
            if (i == j):
                temp.append("x")
            else:
                temp.append(get_dis(lista[i], lista[j]))
        result.append(temp)
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
def write_input_file(n):
    total = np.random.randint(0, n) + 1
    print(total)
    prob = np.random.random_sample()
    lista = []
    for i in range(total):
        temp = get_random_node(1.41*10 ** (9 + 5), 1.41*10 ** (9 + 5))
        lista.append(["index" + str(i)] + temp)
    point_array = get_random_array(total, prob, lista)
    location = [i[0] for i in lista]
    prob2 = np.random.random_sample()
    home = [i for i in range(total) if (not_all_x(point_array[i]) and np.random.random_sample() > prob2)]
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
    for i in range(total):
        for j in range(total - 1):
            f.write(str(point_array[i][j]) + " ")
        f.write(str(point_array[i][total-1]) + "\n")
    return point_array

print(write_input_file(10))
import networkx as nx
import matplotlib.pyplot as plt

INF = 999999999
SCALE = 1000

def show_graph(G):
    print(G.edges())
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def tpp_reduce(input, output):
    fin = open(input, 'r')
    _, _ = int(next(fin)), int(next(fin))
    locations = [''] + next(fin).strip().split(' ') # '' stands for depot
    locations_inv = {locations[i] : i for i in range(len(locations))}
    homes = next(fin).strip().split(' ')
    start = next(fin).strip()
    G = nx.Graph()
    G.add_nodes_from(range(len(locations))) # node 0 is the depot
    G.add_edge(0, locations_inv[start], weight=0) # connect depot to the start
    for i in range(1, len(locations)):
        l = next(fin).strip().split(' ')
        for j in range(i + 1, len(locations)):
            if l[j - 1] != 'x':
                G.add_edge(i, j, weight=int(l[j - 1]))
    fin.close()
    dists = nx.floyd_warshall(G)
    # Change all weights to 2/3 of original
    weights = nx.get_edge_attributes(G, 'weight')
    for w in weights:
        weights[w] *= 2/3
        weights[w] = int(weights[w] * SCALE)
    print(weights)
    nx.set_edge_attributes(G, weights, 'weight')
    data = {'NAME': input,
            'HOMES': homes,
            'LOCATIONS': locations,
            'OFFERS': [[]] + [[(j + 1, dists[i][locations_inv[homes[j]]] * SCALE, 1) for j in range(len(homes))] for i in range(1, len(locations))], # home number, dist to home, quantity
            'G': G}
    return data

def gen_output(data, filename):
    locations = data['LOCATIONS']
    homes = data['HOMES']
    G = data['G']
    fout = open(filename, 'w')
    fout.write('NAME : ' + data['NAME'] + '\n')
    fout.write('TYPE : TPP\n')
    fout.write('DIMENSION : ' + str(len(locations)) + '\n')
    fout.write('EDGE_WEIGHT_TYPE : EXPLICIT\n')
    fout.write('EDGE_WEIGHT_FORMAT : UPPER_ROW\n')

    fout.write('EDGE_WEIGHT_SECTION :\n')
    lines = []
    for v in range(len(locations)):
        neighbors = G[v]
        line = []
        for n in range(v + 1, len(locations)):
            if n in neighbors:
                line.append(str(G[v][n]['weight']))
            else:
                line.append(str(INF))
        lines.append(' '.join(line))
    fout.write('\n'.join(lines))

    fout.write('DEMAND_SECTION :\n')
    fout.write(str(len(homes)) + '\n')
    for i in range(1, len(homes) + 1):
        fout.write(str(i) + ' 1\n')

    fout.write('OFFER_SECTION :\n')
    for i in range(1, len(locations) + 1):
        curr_offers = data['OFFERS'][i - 1]
        line = [i, len(curr_offers)]
        for o in curr_offers:
            line += o
        fout.write(' '.join([str(x) for x in line]) + '\n')
    fout.write('EOF\n')
    fout.close()

data = tpp_reduce('sample.in', 'sample.out')
gen_output(data, 'sample.out')

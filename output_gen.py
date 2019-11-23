import utils
from student_utils import data_parser

def dummy_out(filename):
    """Generates corresponding dummy outputs given input filename
    filename: input filename without .in extension
    """
    input_data = utils.read_file('inputs/' + str(filename) + '.in')
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    fout = open('outputs/' + str(filename) + '.out', 'w')
    fout.write(starting_car_location + '\n')
    fout.write(str(1) + '\n')
    fout.write(' '.join([starting_car_location] + list_houses) + '\n')
    fout.close()
    print('Wrote {}.out'.format(filename))

for p in [50, 100, 200]:
    dummy_out(p)


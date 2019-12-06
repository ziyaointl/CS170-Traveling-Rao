from output_validator import silent_validate_output
from glob import iglob
from os import path
import sys

def get_output_filename(output_folder, name):
    return path.join(output_folder, name+'.out')

def get_input_filename(input_folder, name):
    return path.join(input_folder, name+'.in')

def comp_output(input_folder, output1_folder, output2_folder):
    for f in iglob(output1_folder + '/*.out'):
        output1 = f
        name = f.split('/')[1].split('.')[0]
        output2 = get_output_filename(output2_folder, name)
        input = get_input_filename(input_folder, name)
        cost1 = silent_validate_output(input, output1)
        cost2 = silent_validate_output(input, output2)
        print('Cost for', name, cost1, cost2)

if __name__ == '__main__':
    comp_output(sys.argv[1], sys.argv[2], sys.argv[3])


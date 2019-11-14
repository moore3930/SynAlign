import sys
from collections import *

def main():
    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    input_file_3 = sys.argv[3]
    sent_set_1 = set()
    sent_set_2 = set()
    fout1 = open(input_file_1 + '.s', 'w')
    fout2 = open(input_file_1 + '.t', 'w')
    for line in open(input_file_2):
        sent_set_1.add(line.strip())
    for line in open(input_file_3):
        sent_set_2.add(line.strip())
    for line in open(input_file_1):
        line_array = line.strip().strip('\t')
        if line_array[0] in sent_set_1 and line_array[1] in sent_set_2:
            fout1.write(line_array[0] + '\n')
            fout2.write(line_array[1] + '\n')
    fout1.flush()
    fout2.flush()
    fout1.close()
    fout2.close()
if __name__ == '__main__':
    main()
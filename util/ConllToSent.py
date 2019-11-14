import sys
from collections import *

def main():
    input_file_1 = sys.argv[1]
    fin1 = open(input_file_1)
    fout = open(input_file_1 + '.out', 'w')
    sent_1 = []
    tmp_sent = []
    for line in fin1:
        if line == '\n':
            sent_1.append(' '.join(tmp_sent).replace('-LRB-', '(').replace('-RRB-', ')'))
            tmp_sent = []
        else:
            tmp_sent.append(line.strip().split('\t')[1])

    fin1.close()
    print(len(sent_1))

    for line in sent_1:
        fout.write(line + '\n')

if __name__ == '__main__':
    main()

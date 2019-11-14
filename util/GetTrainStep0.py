import sys

def main():
    input_file = sys.argv[1]
    fin = open(input_file)
    fout = open(input_file + '.fmt', 'w')
    for line in fin:
        line_array = line.strip().split('\t')
        if line_array[0].strip().split(' ')[-1] not in set(['.', '?', '!']):
            continue
        if line_array[1].strip().split(' ')[-1] not in set(['.', '?', '!']):
            continue
        fout.write(line_array[0] + '\t' + line_array[1] + '\n')
    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()

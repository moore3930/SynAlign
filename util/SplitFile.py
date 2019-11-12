
import sys

def main():
    input_file = sys.argv[1]
    output_file_1 = sys.argv[1] + '.s'
    output_file_2 = sys.argv[1] + '.t'
    s_fout = open(output_file_1, 'w')
    t_fout = open(output_file_2, 'w')

    for line in open(input_file):
        line_array = line.strip().split('\t')
        if len(line_array) != 2:
            print(line_array)
            print('Wrong! Musk be source concat target with tab')
            return
        s_fout.write(line_array[0] + '\n')
        t_fout.write(line_array[1] + '\n')
    s_fout.flush()
    t_fout.flush()
    s_fout.close()
    t_fout.close()

if __name__ == '__main__':
    main()

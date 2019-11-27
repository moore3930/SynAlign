import os


def process1():
    de_file = '/Users/Moore/PycharmProjects/SynAlign/data/de-en/de'
    en_file = '/Users/Moore/PycharmProjects/SynAlign/data/de-en/en'
    output_file = '/Users/Moore/PycharmProjects/SynAlign/data/de-en/de-en-test.txt'
    fout = open(output_file, 'w')
    for de, en in zip(open(de_file, encoding='cp1252'), open(en_file, encoding='cp1252')):
        line = de.strip() + '\t' + en.strip() + '\n'
        fout.write(line)
    fout.flush()
    fout.close()

def process2():
    align_file = '/Users/Moore/PycharmProjects/SynAlign/data/de-en/alignmentDeEn'
    output_file = '/Users/Moore/PycharmProjects/SynAlign/data/de-en/de-en-test-wa.txt'
    fout = open(output_file, 'w')
    index = 1
    for line in open(align_file, encoding='cp1252'):
        line = line.strip()
        if len(line) == 0:
            continue
        line_array = line.strip().split(' ')
        if len(line_array) == 2 and line_array[0] == 'SENT:':
            index = int(line_array[1]) + 1
            continue
        elif len(line_array) == 3:
            if line_array[0] == 'S':
                fout.write("num-{} {} -> {} S\n".format(index, int(line_array[1])+1, int(line_array[2]) + 1))
            elif line_array[0] == 'P':
                fout.write("num-{} {} -> {} P\n".format(index, int(line_array[1]) + 1, int(line_array[2]) + 1))
            else:
                print(line_array)
        else:
            print(line_array)
    fout.flush()
    fout.close()

process2()



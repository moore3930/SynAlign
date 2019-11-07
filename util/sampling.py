import random
path = '../data/en-fr.txt'
path_out = '../data/en-fr-sample.txt'

fout = open(path_out, 'w')
for line in open(path):
    if random.random() < 0.05:
        line = line.strip()
        fout.write(line + '\n')
fout.flush()
fout.close()

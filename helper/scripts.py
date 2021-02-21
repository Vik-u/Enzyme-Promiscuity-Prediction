import numpy as np


def read_fasta(filename):
    names = []
    sequences = []
    with open(filename,'r') as f:
        for lines in f:
            if lines.startswith('>'):
                names.append(lines.strip().replace('>',''))
            else:
                sequences.append(lines.strip())
    return names,sequences


def read_pssm_profile(filename):
    matrix = []
    with open(filename,'r') as f:
        all_lines = f.readlines()
        for i,line in enumerate(all_lines):
            if i>2 and i<(len(all_lines)-6):
                matrix.append(line.strip().split())
    return np.array(matrix)
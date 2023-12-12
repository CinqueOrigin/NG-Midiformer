import os
import random
import pandas as pd
from tqdm import tqdm

def cut(input_file,frequency, outfile):
    num =1
    with open(outfile,'w') as fout:
        with open(input_file,'r') as fin:
            for line in tqdm(fin):
                freq = line.split(',')[-1]
                if int(freq) <frequency:
                    break
                fout.write(line)
                num+=1
    print(num)


if __name__ == '__main__':
    cut('~/ngram800.txt',2200,'~/ngram2200.txt')
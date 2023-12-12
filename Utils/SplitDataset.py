import pandas as pd
import json
import random
def splitdataset(input_file):
    with open(input_file, 'r') as f:
        data = f.readlines()
    random.shuffle(data)
    n = len(data)
    n1 = int(n * 0.9)
    data1 = data[:n1]
    data2 = data[n1:]

    with open(input_file.replace('.json','_train.json'),'w') as f:
        f.writelines(data1)

    with open(input_file.replace('.json','_valid.json'),'w') as f:
        f.writelines(data2)

if __name__ == '__main__':
    splitdataset('~/dataSet/genre/UnicodesCP/mergeUnicodeFilePianoGenreMel.json')
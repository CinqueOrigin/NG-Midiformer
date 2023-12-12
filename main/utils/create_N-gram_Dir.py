import os
import random
import pandas as pd
from tqdm import tqdm
import multiprocessing
from subprocess import run
import re
OUT_NAME='ngram.txt'

def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if not file.startswith('midi_train_'):continue
            mix_path = os.path.join(root, file)
            pure_path = mix_path[len(words_dir)+1:]
            file_list.append(pure_path)
    return file_list

def run_cmd(cmd_str=''):
    # cmd_str = 'echo 2'
    run(cmd_str,shell=True)

def Create_N_gram_dir(input_dir):
    with multiprocessing.Pool(processes=2) as pool:
        file_list = traverse_words_dir_recurrence(input_dir)
        
        output_dir = './ngram_temp'
        cmd_Str_list=[]
        for file in file_list:
            index = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)][0]
            index = int(index)
            if index == str(0):continue
            output_file = 'ngram'+str(index)+'.txt'
            str_temp = 'python CreateN-gram.py --input_file '+os.path.join(input_dir,file)+' --output_dir '+output_dir+' --outputfile_Name '+output_file
            print(str_temp)
            cmd_Str_list.append(str_temp)   
        pool.map(run_cmd,cmd_Str_list)
        pool.close()
        pool.join()



Create_N_gram_dir('~/workSpace/musicbert/SuboutputOctuple_30000')
import os
import random
import pandas as pd
from tqdm import tqdm
from subprocess import run
import multiprocessing
import argparse
import re
OUT_NAME='ngram.txt'

def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            mix_path = os.path.join(root, file)
            pure_path = mix_path[len(words_dir)+1:]
            file_list.append(pure_path)
    return file_list

def run_cmd(cmd_str=''):
    # cmd_str = 'echo 2'
    run(cmd_str,shell=True)

def create_N_gram(input_file,N=3,output_dir='.',outputfile_Name=None,trim=0):
    print('dealing with '+input_file)
    with open(input_file,'r') as fin:
        corpus={}
        #1.read each line,and split it into tokens
        for line in tqdm(fin):
            line = line.strip('\n')
            tokens = line.split(' ')
        #2.get the frequency of N-gram for N in 1,...,N
            for num in range(2,N+1):
                for i in range(len(tokens)-num+1):
                    key = ' '.join(tokens[j] for j in range(i,i+num))
                    if(corpus.get(key) is None):
                        corpus[key] = 1
                    else:
                        corpus[key]+=1
                
        #3.sort and save as outputfile
        corpus = sorted(corpus.items(),key=lambda x: -x[1])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if outputfile_Name is None:
            outputfile_Name = OUT_NAME
        outputfile =os.path.join(output_dir,outputfile_Name)
        with open(outputfile,'w') as fout:
            for key,value in corpus:
                if(value<trim):
                    break
                fout.write(key+','+str(value)+'\n')

def Create_N_gram_dir(input_dir,output_dir=None):
    with multiprocessing.Pool(processes=2) as pool:
        file_list = traverse_words_dir_recurrence(input_dir)
        if output_dir is None:
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

def Trim(input_file,output_file,num=200):
    with open(input_file,'r') as fin:
        with open(output_file,'w') as fout:
            for line in fin:
                freq = int(line.split(',')[-1])
                if freq < num:
                    break
                fout.write(line)

def mergeNgram(input_dir,outputFile,num=450):
    file_list = traverse_words_dir_recurrence(input_dir)
    corpus = {}
    with open(outputFile,'w') as fout:
        for file in tqdm(file_list):
            if file.startswith('ngram'):
                with open(os.path.join(input_dir,file),'r') as fin:
                    for line in fin:
                        key,freq = line.split(',')
                        freq = int(freq)
                        # print(key)
                        if(corpus.get(key) is not None):
                            corpus[key]+=freq
                        else:
                            corpus[key] = freq
        corpus = sorted(corpus.items(),key=lambda x: -x[1])
        for key,value in corpus:
            print(key,value)
            if(value<num):
                break
            fout.write(key+','+str(value)+'\n')

def mergeVocab(input_dir,outputFile,num=450):
    file_list = traverse_words_dir_recurrence(input_dir)
    print(file_list)
    corpus = {}
    with open(outputFile,'w') as fout:
        for file in tqdm(file_list):
            if not file.startswith('codes_30000_Octuple_'):continue
            with open(os.path.join(input_dir,file),'r') as fin:
                for line in fin:
                    key = line.split(' ')[:-1]
                    key = ' '.join(it for it in key)
                    freq = line.split(' ')[-1]
                    # print(key)
                    # print(freq)
                    # exit()
                    freq = int(freq)
                    # print(key)
                    if(corpus.get(key) is not None):
                        corpus[key]+=freq
                    else:
                        corpus[key] = freq
        corpus = sorted(corpus.items(),key=lambda x: -x[1])
        for key,value in corpus:
            print(key,value)
            if(value<num):
                break
            fout.write(key+' '+str(value)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply fastBPE')
    parser.add_argument('--input_file',default='./../../dataSet/FinalDataSet/unicodesCP/mergeUnicodeFiles.txt', help='words path')
    parser.add_argument('--output_dir',default='./../../dataSet/FinalDataSet/subUnicodesCP',help='output path')
    parser.add_argument('--outputfile_Name',default='./../../dataSet/FinalDataSet/unicodesCP/codes',help='output path')
    args = parser.parse_args()
    create_N_gram('~/dataSet/PianoDataset/cp4/uc4/mergeUnicodeFilePianoBped.txt',N=4,output_dir='~/dataSet/PianoDataset/cp4/uc4',
    outputfile_Name='ngram.txt',trim=400)
    # Trim('~/workSpace/musicbert/lmd_full_data_raw/ngram.txt','~/workSpace/musicbert/lmd_full_data_raw/ngram1000.txt',num=6000)
    # create_N_gram(args.input_file,N=3,
    #     output_dir=args.output_dir,outputfile_Name=args.outputfile_Name,trim=100)
    # mergeNgram('~/workSpace/musicbert/output/subTrainBped','~/workSpace/musicbert/output/subTrainBped/ngram_30000_800.txt',1600)
    # mergeVocab('~/workSpace/musicbert/outputOctuple','~/workSpace/musicbert/outputOctuple/codes_30000_octuple.txt',500)
    # Trim('~/dataSet/outputSubUnicode/mergedTrainOctuple.txt','~/workSpace/ZEN-2_Pretrain/4gramTrim400ctuple.txt')

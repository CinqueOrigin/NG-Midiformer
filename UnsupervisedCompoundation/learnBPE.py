import os
from tqdm import tqdm
import fastBPE
from subprocess import run
import argparse
import re

def traverse_words_dir(words_dir):
    """
    从words文件夹中获取所有的words文件名称
    :param words_dir:
    :return:
    """
    file_names=[]
    for f in os.listdir(words_dir):
        if f.endswith(".txt"):
            file_names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    return file_names

def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(".txt"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    return file_list

def run_cmd(cmd_str=''):
    run(cmd_str,shell=True)

def learnBpe(inputfile,outputfile,Ncodes=200):
    learn_str='./fastBPE/fast learnbpe '+str(Ncodes)+' '+inputfile+'>'+outputfile
    run_cmd(learn_str)

def learnBpeWithForDir(inputdir,outputfile,Ncodes=200):
    file_list = traverse_words_dir_recurrence(inputdir)
    fileStr=''
    for idx,file in enumerate(file_list):
        index = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)][0]
        index = int(index)
        if index == str(0):continue
        print(file)
        fileStr+= (str(os.path.join(inputdir,file)) + ' ')
        if(idx == 0):break
    learn_str='./fastBPE/fast learnbpe '+str(Ncodes)+' '+fileStr+'> '+outputfile+'.txt'
    # learn_str ='cat '+fileStr+' | ./fastBPE/fast learnbpe '+str(Ncodes)+' - > '+outputfile+'.txt'
    print(learn_str)   
    run_cmd(learn_str)

def mergeBPEfile(input_fir,output_file,prefix):
    file_list =  traverse_words_dir(input_fir)
    with open(output_file,'w') as fout:
        for file in file_list:
            if file.startswith(prefix):
                with open(file+'.txt') as fin:
                    for line in fin:
                        fout.write(line)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='apply fastBPE')
    parser.add_argument('--txt_file',default='~/workSpace/musicbert/output/subTrain', help='words path')
    parser.add_argument('--outputCodes_file',default='~/workSpace/musicbert/output/codes_3000',help='output path')
    parser.add_argument('--vocab_size',default=5000,help='vocab size')
    args = parser.parse_args()
    learnBpe(args.txt_file,
    args.outputCodes_file,
    args.vocab_size)
    # learnBpeWithForDir(args.txt_file,
    # args.outputCodes_file,
    # args.vocab_size)
    # mergeBPEfile('~/workSpace/musicbert/outputOctuple','',)

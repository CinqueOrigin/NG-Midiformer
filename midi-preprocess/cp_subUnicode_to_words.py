from codecs import unicode_escape_decode
import os
import pickle
import argparse
import collections
from unittest.mock import _patch_dict
from tqdm import tqdm
import numpy as np

def traverse_words_dir(subUnic_dir):
    """
    从words文件夹中获取所有的words文件名称
    :param subUnic_dir:
    :return:
    """
    file_names=[]
    for f in os.listdir(subUnic_dir):
        if f.endswith(".txt"):
            file_names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    return file_names

def traverse_words_dir_recurrence(subUnic_dir):
    file_list = []
    for root, h, files in os.walk(subUnic_dir):
        for file in files:
            if file.endswith(".txt"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(subUnic_dir)+1:]
                file_list.append(pure_path)
    return file_list

def covert_sub_unicodes_to_cp_words(subUnic_dir,dictionary_path,output_dir):
    subUnic_files = traverse_words_dir_recurrence(subUnic_dir)

    word2unic,unic2word = pickle.load(open(dictionary_path,'rb'))

    for file in tqdm(subUnic_files):
        Words=[]
        with open(os.path.join(subUnic_dir,file)) as f:
            for line in f:
                line=line.replace('\n','')
                Word=[]
                for unicGroup in line.split('@@ '):
                    word=[]
                    for unic in unicGroup:
                        ch=unic2word[unic]
                        word.append(ch)
                    Word.append(word)
                if unic2word[line[0]] =='Note':Word=['000']+Word
                elif unic2word[line[0]] =='EOS':Word=['000']+Word+['000']
                else :Word=Word+['000']
                Words.append(Word)
            
            path_outfile = os.path.join(output_dir, file)
            fn = os.path.basename(path_outfile)
            os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
            with open(path_outfile,'w') as fout:
                fout.write("\n".join(["@@".join([" ".join(str(ch) for ch in word) for word in Word]) for Word in Words])+"\n")
            exit()

if __name__ =='__main__':
    covert_sub_unicodes_to_cp_words('~/dataSet/MAESTRO/subUnicodesCP','~/dataSet/MAESTRO/unicodesCP/dictionary.pkl','~/dataSet/MAESTRO/RegeneratedWordsCP')
    


        
                    


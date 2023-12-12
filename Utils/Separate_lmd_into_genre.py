from codecs import unicode_escape_decode
import os
import pickle
import argparse
import collections
from unittest.mock import _patch_dict
from tqdm import tqdm
import numpy as np
import json

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

def traverse_words_dir_recurrence(words_dir,input_suffix='.mid'):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(input_suffix):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    # print(file_list)
    return file_list

def convert_CpWords_to_Unicode(words_dir,output_dir,dic_file,baseNum=1000):
    #1.get all filenames
    word_files=traverse_words_dir_recurrence(words_dir)
    # print(len(word_files))
    #2.init data structure
    if dic_file is None:
        word2unic={}
        unic2word={}
        word2unic=word2unic.fromkeys([0,1,2,3,4,5,6],{})
        unic2word=unic2word.fromkeys([0,1,2,3,4,5,6],{})
        print(word2unic.keys())
        index=baseNum
        #3.allocate unicode to a cpWord
        cnt=0
        for file in tqdm(word_files):
            Words=[]
            with open(os.path.join(words_dir,file)) as f:
                for line in f:
                    line = line.replace('\n','')
                    # print(line)
                    word=line.split(' ')
                    unic=''
                    if word[3] == 'Metrical':
                        for idx,item in enumerate(word):
                            if idx>3 :break
                            if not word2unic[idx].get(item):
                                indexTemp=hex(index)
                                indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
                                indexTemp=indexTemp.decode('unicode_escape')
                                # print(indexTemp)
                                word2unic[idx][item]=indexTemp
                                unic2word[indexTemp]=item
                                index=index+1
                                unic+=indexTemp
                                cnt=cnt+1
                            else:
                                unic+=word2unic[idx][item]
                        Words.append(unic)
                    elif word[3] == 'Note':
                        for idx,item in enumerate(word):
                            if idx<3 :continue
                            if not word2unic[idx].get(item):
                                indexTemp=hex(index)
                                indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
                                indexTemp=indexTemp.decode('unicode_escape')
                                word2unic[idx][item]=indexTemp
                                unic2word[indexTemp]=item
                                index=index+1
                                unic+=indexTemp
                                cnt=cnt+1
                            else:
                                unic+=word2unic[idx][item]
                        Words.append(unic)
                    elif word[3] == 'EOS':
                        if not word2unic[3].get('EOS'):
                            indexTemp=hex(index)
                            indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
                            indexTemp=indexTemp.decode('unicode_escape')
                            # print(indexTemp)
                            word2unic[3]['EOS'] = indexTemp
                            unic2word[indexTemp]='EOS'
                            index=index+1
                            unic+=indexTemp
                            cnt=cnt+1
                        else:
                            unic+=word2unic[3]['EOS']
                        Words.append(unic)
                    else:raise('No Type!')

            path_outfile = os.path.join(output_dir, file)
            fn = os.path.basename(path_outfile)
            os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
            with open(path_outfile,'w') as fout:
                fout.write("\n".join(["".join([str(it) for it in item]) for item in Words])+"\n")

    else:
        word2unic,unic2word = pickle.load(open(dic_file,'rb'))
        index=baseNum
        #3.allocate unicode to a cpWord
        cnt=0
        for file in tqdm(word_files):
            Words=[]
            with open(os.path.join(words_dir,file)) as f:
                for line in f:
                    line = line.replace('\n','')
                    # print(line)
                    word=line.split(' ')
                    unic=''
                    if word[3] == 'Metrical':
                        for idx,item in enumerate(word):
                            if idx>3 :break
                            if not word2unic[idx].get(item):
                                print('error:no such word:'+item+'in this dictionary')
                                exit()
                            else:
                                unic+=word2unic[idx][item]
                        Words.append(unic)
                    elif word[3] == 'Note':
                        for idx,item in enumerate(word):
                            if idx<3 :continue
                            if not word2unic[idx].get(item):
                                print('error:no such word:'+item+'in this dictionary')
                            else:
                                unic+=word2unic[idx][item]
                        Words.append(unic)
                    elif word[3] == 'EOS':
                        if not word2unic[3].get('EOS'):
                            print('error:no such word:'+item+'in this dictionary')
                        else:
                            unic+=word2unic[3]['EOS']
                        Words.append(unic)
                    else:raise('No Type!')

            path_outfile = os.path.join(output_dir, file)
            fn = os.path.basename(path_outfile)
            os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
            with open(path_outfile,'w') as fout:
                fout.write("\n".join(["".join([str(it) for it in item]) for item in Words])+"\n")

    print('num_unicodes:',cnt)
    if dic_file is None:
        path_dict=os.path.join(output_dir,'dictionary.pkl')
        pickle.dump((word2unic, unic2word),open(path_dict, 'wb'))

def merge_lmd_into_Genre_into_json(input_dir,output_dir=None,input_suffix='.txt',json_file=None):
    if output_dir is None:
        output_dir = input_dir
    #1.get file list
    file_list =  traverse_words_dir_recurrence(input_dir,input_suffix)
    #2.read map jason to a dict
    labels = dict()
    subsets=['masd','topmagd']
    with open(json_file,'r') as f_label:
        data = json.load(f_label)
        for subset in subsets:
            labels[subset]=dict()
            for s in data[subset].items():
                labels[subset][s[0]] = tuple(
                    sorted(set(i.strip().replace(' ', '-') for i in s[1])))
    # print(labels['masd'].keys())       
    #3.convert each file in the file list into a specific file with its corresponding labels
    for subset in subsets:
        file_list = traverse_words_dir_recurrence(os.path.join(input_dir,subset),input_suffix='.txt')
        fn_out = subset+'merged.json'
        with open(os.path.join(output_dir,fn_out),'w') as f_out:
            for file in tqdm(file_list):
                id = file.split('/')[-1].split('.')[0]
                # print(id)
                # print(file)
                label_str = labels[subset][id]
                label_str = ','.join(id for id in label_str)
                # print(label_str)
                with open(os.path.join(input_dir,subset,file)) as f_in:
                    line = f_in.readline().strip()
                    f_out.write(('{"text":"'+line)+'","label":"'+label_str+'"}\n')


def Separate_lmd_into_Genre_to_dir(input_dir,output_dir=None,input_suffix='.txt',json_file=None):
    if output_dir is None:
        output_dir = input_dir
    #1.get file list
    file_list =  traverse_words_dir_recurrence(input_dir,input_suffix)
    #2.read map jason to a dict
    labels = dict()
    subsets=['masd','topmagd']
    with open(json_file,'r') as f_label:
        data = json.load(f_label)
        for subset in subsets:
            labels[subset]=dict()
            for s in data[subset].items():
                labels[subset][s[0]] = tuple(
                    sorted(set(i.strip().replace(' ', '-') for i in s[1])))
    # print(labels.keys())       
    #3.convert each file in the file list into a specific file with its corresponding labels
    for subset in subsets:
        out_dir=os.path.join(output_dir,subset)
        os.system('mkdir -p {}'.format(out_dir))
        for file in file_list:
            id = file.split('/')[-1].split('.')[0]
            if id in labels[subset]:
                with open(os.path.join(input_dir,file)) as fin,open(os.path.join(out_dir,subset,file)) as f_out:
                    for line in fin:
                        line = line.strip()
                        label_str = ','.join(str(label) for label in labels[subset][id])
                        f_out.write(('{"text":"'+line)+'","label":"'+label_str+'"}\n')

def Separate_lmd_into_Genre_with_only_file(input_dir,output_dir=None,input_suffix='.txt',json_file=None):
    if output_dir is None:
        output_dir = input_dir
    #1.get file list
    file_list =  traverse_words_dir_recurrence(input_dir,input_suffix)
    #2.read map jason to a dict
    labels = dict()
    subsets=['masd','topmagd']
    with open(json_file,'r') as f_label:
        data = json.load(f_label)
        for subset in subsets:
            labels[subset]=dict()
            for s in data[subset].items():
                labels[subset][s[0]] = tuple(
                    sorted(set(i.strip().replace(' ', '-') for i in s[1])))
    # print(labels.keys())       
    #3.convert each file in the file list into a specific file with its corresponding labels
    for subset in subsets:
        out_dir=os.path.join(output_dir,subset)
        os.system('mkdir -p {}'.format(out_dir))
        for file in tqdm(file_list):
            id = file.split('/')[-1].split('.')[0]
            if id in labels[subset]:
                os.system('mkdir -p {}'.format(os.path.join(out_dir,file.split('/')[0])))
                with open(os.path.join(input_dir,file),'r') as fin,open(os.path.join(out_dir,file),'w') as f_out:
                    for line in fin:
                        f_out.write(line)

def testUnicode():
    base=1000
    end=1006
    for index2 in range(base,end):
        index = hex(index2)
        print(str(index).encode())
        print(str(index2).encode())
        c = (b"\u"+str(index).encode()).replace(b'x',b'')
        c2 = b"\u"+str(index2).encode()
        d = c.decode('utf8')
        d2 = c2.decode('utf8')
        e = c.decode('unicode_escape')
        e2 = c2.decode('unicode_escape')
        print(c,' ',d,' ',e)
        print(c2,' ',d2,' ',e2)


if __name__ == '__main__':
    # parseToChars('./../../dataSet/MAESTRO/wordsCP','./temp')
    parser = argparse.ArgumentParser(description='Process the cp events file to words.')
    parser.add_argument('--words_path',default='./../../dataSet/MAESTRO/wordsCP', help='words path')
    parser.add_argument('--output_path',default='./../../dataSet/MAESTRO/unicodesCP',help='output path')
    parser.add_argument('--dict_file',default=None,help='the position of dictionary.pkl')
    parser.add_argument('--baseNum', default=1000, help='the start of Unicode encoding')
    args = parser.parse_args()
    # traverse_words_dir_recurrence('~/dataSet/lmd/lmd_full')
    merge_lmd_into_Genre_into_json('~/dataSet/lmd/wordsCPsubsetsUnicodeBped','~/dataSet/lmd/wordsCPsubsetsUnicodeBped','.txt','~/dataSet/lmd/midi_genre_map.json')
    # convert_CpWords_to_Unicode(args.words_path,args.output_path,args.dict_file,args.baseNum)



    
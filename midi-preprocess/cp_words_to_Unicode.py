from codecs import unicode_escape_decode
import os
import pickle
import argparse
import collections
from unittest.mock import _patch_dict
from tqdm import tqdm
import numpy as np


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


def convert_CpWords_to_Unicode_2_oneFile(words_dir,output_dir,dic_file,baseNum=1000):
    #1.get all filenames
    word_files=traverse_words_dir_recurrence(words_dir)
    # print(len(word_files))
    #2.init data structure
    word2unic,unic2word = pickle.load(open(dic_file,'rb'))
    index=baseNum
    #3.allocate unicode to a cpWord
    cnt=0
    for file in tqdm(word_files):
        Words=[]
        with open(os.path.join(words_dir,file)) as f_in:
            for line in f_in:
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
                fout.write(" ".join(["".join([str(it) for it in item]) for item in Words])+"\n")

    print('num_unicodes:',cnt)
    if dic_file is None:
        path_dict=os.path.join(output_dir,'dictionary.pkl')
        pickle.dump((word2unic, unic2word),open(path_dict, 'wb'))

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
    convert_CpWords_to_Unicode_2_oneFile(args.words_path,args.output_path,args.dict_file,args.baseNum)



    
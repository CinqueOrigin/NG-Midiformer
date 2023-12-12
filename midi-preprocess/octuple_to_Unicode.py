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


def traverse_words_dir_recurrence_down(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith("test.txt") or file.endswith("train.txt"):
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

def convert_octupleMIDI_to_unicode_for_whole_word(input_file,output_file,dic_file,baseNum=1000):
    with open(input_file,'r') as fin:
        with open(output_file,'w') as fout:
            if dic_file is None:
                word2unic={}
                unic2word={}
                word2unic=word2unic.fromkeys([0,1,2,3,4,5,6,7],{})
                unic2word=unic2word.fromkeys([0,1,2,3,4,5,6,7],{})
                print(word2unic.keys())
                index=baseNum
                cnt = 0
                for line in tqdm(fin):
                    Words=[]
                    line = line.replace('\n','')
                    word=line.split(' ')[8:-7]
                    # print(word)
                    for idx,item in enumerate(word):
                        idx = idx%8
                        if(idx == 0):unic = ''
                        item = item.replace('<','').replace('>','')
                        tokenId =item.split('-')[-1]
                        tokenId=int(tokenId)
                        if not word2unic[idx].get(tokenId):
                            indexTemp=hex(index)
                            indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
                            indexTemp=indexTemp.decode('unicode_escape')
                            # print(indexTemp)
                            word2unic[idx][tokenId]=indexTemp
                            unic2word[indexTemp]=tokenId
                            index=index+1
                            unic+=indexTemp
                            cnt=cnt+1
                        else:
                            unic+=word2unic[idx][tokenId]
                        if idx == 7:
                            Words.append(unic)
                            
                    fout.write(" ".join(["".join([str(it) for it in item]) for item in Words])+"\n")
            else:
                word2unic,unic2word = pickle.load(open(dic_file,'rb'))
                index=baseNum
                cnt = 0
                for line in tqdm(fin):
                    Words=[]
                    line = line.replace('\n','')
                    word=line.split(' ')[8:-7]
                    # print(word)
                    for idx,item in enumerate(word):
                        idx = idx%8
                        if(idx == 0):unic = ''
                        item = item.replace('<','').replace('>','')
                        tokenId =item.split('-')[-1]
                        tokenId=int(tokenId)
                        if not word2unic[idx].get(tokenId):
                            print('error:no such word:'+item+'in this dictionary')
                            exit()
                        else:
                            unic+=word2unic[idx][tokenId]
                        if idx == 7:
                            Words.append(unic)
                            
                    fout.write(" ".join(["".join([str(it) for it in item]) for item in Words])+"\n")
    print('num_unicodes:',cnt)
    if dic_file is None:
        pickle.dump((word2unic, unic2word),open(os.path.join(os.path.dirname(output_file),'dictionary.pkl'),'wb'))

def convert_octupleMIDI_to_unicode_nsp_for_whole_word(input_file,output_file,dic_file,baseNum=1000):
    with open(input_file,'r') as fin:
        with open(output_file,'w') as fout:
            if dic_file is None:
                word2unic={}
                unic2word={}
                word2unic=word2unic.fromkeys([0,1,2,3,4,5,6,7],{})
                unic2word=unic2word.fromkeys([0,1,2,3,4,5,6,7],{})
                print(word2unic.keys())
                index=baseNum
                cnt = 0
                for line in tqdm(fin):
                    Words=[]
                    line = line.replace('\n','')
                    word=line.split(' ')[8:-7]
                    # print(word)
                    position = word.index('<0--10>')
                    for idx,item in enumerate(word):
                        idx = idx%8
                        if(idx == 0):unic = ''
                        item = item.replace('<','').replace('>','')
                        tokenId =item.split('-')[-1]
                        tokenId=int(tokenId)
                        if not word2unic[idx].get(tokenId):
                            indexTemp=hex(index)
                            indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
                            indexTemp=indexTemp.decode('unicode_escape')
                            # print(indexTemp)
                            word2unic[idx][tokenId]=indexTemp
                            unic2word[indexTemp]=tokenId
                            index=index+1
                            unic+=indexTemp
                            cnt=cnt+1
                        else:
                            unic+=word2unic[idx][tokenId]
                        if idx == 7:
                            Words.append(unic)
                            
                    fout.write(" ".join(["".join([str(it) for it in item]) for item in Words])+"\n")
            else:
                word2unic,unic2word = pickle.load(open(dic_file,'rb'))
                index=baseNum
                cnt = 0
                for line in tqdm(fin):
                    Words=[]
                    line = line.replace('\n','')
                    word=line.split(' ')[8:-7]
                    # print(word)
                    position = word.index('<0--10>')
                    # print(word)
                    # print(word[0:position])
                    # print(word[position+1:])
                    # exit()
                    for idx,item in enumerate(word[0:position]):
                        idx = idx%8
                        if(idx == 0):unic = ''
                        item = item.replace('<','').replace('>','')
                        tokenId =item.split('-')[-1]
                        tokenId=int(tokenId)
                        if not word2unic[idx].get(tokenId):
                            print('error:no such word:'+item+'in this dictionary')
                            exit()
                        else:
                            unic+=word2unic[idx][tokenId]
                        if idx == 7:
                            Words.append(unic)
                            
                    fout.write(" ".join(["".join([str(it) for it in item]) for item in Words])+"")
                    fout.write(' ### ')
                    for idx,item in enumerate(word[position+1:]):
                        idx = idx%8
                        if(idx == 0):unic = ''
                        item = item.replace('<','').replace('>','')
                        tokenId =item.split('-')[-1]
                        tokenId=int(tokenId)
                        if not word2unic[idx].get(tokenId):
                            print('error:no such word:'+item+'in this dictionary')
                            exit()
                        else:
                            unic+=word2unic[idx][tokenId]
                        if idx == 7:
                            Words.append(unic)
                            
                    fout.write(" ".join(["".join([str(it) for it in item]) for item in Words])+"")
    print('num_unicodes:',cnt)
    if dic_file is None:
        pickle.dump((word2unic, unic2word),open(os.path.join(os.path.dirname(output_file),'dictionary.pkl'),'wb'))


def convert_octupleMIDI_to_unicode_for_single_element(input_file,output_file,dic_file):
    with open(input_file,'r') as fin:
        out_dir = os.path.dirname(output_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(output_file,'w') as fout:
            if dic_file is None:
                raise('error! no dict file!')
            else:
                octuple2unicode={}
                with open(dic_file,'r') as f_dict:
                    for line in f_dict:
                        line = line.strip()
                        octupleElement,unic=line.split(' ')
                        octuple2unicode[octupleElement] = unic
                print(octuple2unicode)
                cnt = 0
                for line in tqdm(fin):
                    Words=[]
                    line = line.replace('\n','')
                    words=line.split(' ')
                    for word in words:
                        Words.append(octuple2unicode[word])
                    fout.write(" ".join([str(item) for item in Words])+"\n")
    

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

def convert_octuple_to_unicode_dir(input_file,dic_file,baseNum=1000):
    file_list = traverse_words_dir_recurrence_down(input_file)
    for file in file_list:
        out_file = file[:-4]+'_unicode.txt'
        print(out_file)
        convert_octupleMIDI_to_unicode_for_whole_word(os.path.join(input_file,file),os.path.join(input_file,out_file),
        dic_file,baseNum)


if __name__ == '__main__':
    # parseToChars('./../../dataSet/MAESTRO/wordsCP','./temp')
    parser = argparse.ArgumentParser(description='Process the cp events file to words.')
    parser.add_argument('--input_file',default='~/workSpace/musicbert/masd_data_raw/4/train.txt', help='words path')
    parser.add_argument('--output_file',default='~/workSpace/musicbert/masd_data_raw/4/train_unicode.txt',help='output path')
    parser.add_argument('--dict_file',default='~/workSpace/musicbert/output/dictionary.pkl',help='the position of dictionary.pkl')
    parser.add_argument('--baseNum', default=1000, help='the start of Unicode encoding')
    args = parser.parse_args()
    # print('test')
    # convert_CpWords_to_Unicode(args.input_file,args.output_file,args.dict_file,args.baseNum)
    # convert_octupleMIDI_to_unicode_for_whole_word(args.input_file,args.output_file,args.dict_file,args.baseNum)
    convert_octupleMIDI_to_unicode_for_single_element('~/workSpace/musicbert/topmagd_data_raw/0/test.txt',
                                                      '~/workSpace/musicbert/topmagd_data_raw/0/testSingle.txt',
                                                      '~/workSpace/musicbert/lmd_full_data_raw/vocabdict.txt')
    



    
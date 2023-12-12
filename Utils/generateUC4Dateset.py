import os
import random
from tqdm import tqdm
import pickle
import numpy as np
import re
import argparse
# DEFAULT_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])  

def traverse_words_dir_recurrence(words_dir,input_suffix='.txt'):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(input_suffix):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    # print(file_list)
    return file_list

def getTrain_and_Value_DataSet_for_sequence(input_dir):
    txts = traverse_words_dir_recurrence(os.path.join(input_dir,'subtxt'))
    labels = traverse_words_dir_recurrence(os.path.join(input_dir,'label'))
    # print(labels)
    data_list=[]
    for idx,file in tqdm(enumerate(txts)):
        # print(root,h,file)
        # print(file)
        fn = file.split('.')[0]+'.label'
        label_file = open(os.path.join(input_dir,'label',fn),'r')
        label=int(label_file.readline())
        # print(label)
        # print(label)

        with open(os.path.join(input_dir,'subtxt',file),'r') as f:
            line = f.readline()
            line = line.replace('\n','')
        data_list.append([line,label])
    return data_list

def cut_list(raw_list, ratio=0.05):
    '''
    raw_list:the list to be split
    ratio：the ratio for one and another new list
    return:two lists,the first with len*ratio,the latter with the others
    '''
    length = len(raw_list)
    val_idx = random.sample(range(length), int(length*ratio)) 
    train = []             
    val = []               
 
    index = [False for _ in range(length)]  # record the valid index
    for i in val_idx:
        index[i] = True
 
    # start appending
    for i, element in enumerate(index):
        if element == True:
            val.append(raw_list[i])
        else:
            train.append(raw_list[i])
    return val, train

def reshape(lst, n):
    lst = lst.split(' ')
    end = len(lst)//n
    if end * n == len(lst):
        return [lst[i*n:(i+1)*n] for i in range(end)]
    else:
        return [lst[i*n:(i+1)*n] for i in range(end)]+[lst[end*n:(end+1)*n]]

def reshape_with_cp(lst,n):
    lst = lst.split(' ')
    end = len(lst)//n
    res=[]
    count = 0
    tmp=[]
    for item in lst:
        if item[-2:] == '@@':
            tmp.append(item)
        else:
            tmp.append(item)
            count+=1
            if(count == n):
                count=0
                res.append(tmp)
                tmp=[]
    if(count != n):res.append(tmp)
    return res

def reshape_with_cp_and_label(lst,labellst,n):
    # print(len(lst.split(' ')))
    # print(len(labellst))
    lst = lst.split(' ')
    # end = len(lst)//n
    res=[]
    res_label=[]
    count = 0
    tmp=[]
    tmp_label=[]
    for idx,item in enumerate(lst):
        if item[-2:] == '@@':
            tmp.append(item)
            tmp_label.append(labellst[idx])
        else:
            tmp.append(item)
            tmp_label.append(labellst[idx])
            count+=1
            if(count == n):
                count=0
                res.append(tmp)
                res_label.append(tmp_label)
                tmp=[]
                tmp_label=[]
    if(count != n):
        res.append(tmp)
        res_label.append(tmp_label)

    return res,res_label

def generate_json_file(inputfile,outfile=None,max_sequence_length=512):
    if outfile is None:
        outfile = inputfile.split('.')[0]+str(max_sequence_length)+'.json'
    with open(inputfile,'r') as fin:
        with open(os.path.join('.',outfile),'w') as fout:
            for line in tqdm(fin):
                line=line.replace('\n','')
                #reconstruct each line
                temp = reshape(line,max_sequence_length-2)
                for newline in temp:
                    # the reason of -2 is due to the [CLS] and [SEP]
                    # newline = reshape(line,max_sequence_length-2)
                    # print(newline,len(newline))
                    jsonLine = '{"text":"'+' '.join(element for element in newline)+'"}'+'\n'
                    fout.write(jsonLine)
                    # print(len(newline))

def generate_json_file_for_classification_from_list(raw_list,output_file,max_sequence_length=512):
    with open(output_file,'w') as fout:
        for sentence,label in raw_list:
            newlines = reshape(sentence,max_sequence_length)
            for line in newlines:
                jsonLine = '{"text":"'+sentence+'","label":'+str(label)+'}'
                # print(jsonLine)
                fout.write(jsonLine+'\n')

def generate_json_file_for_classification_from_list_Long_with_BPE(raw_list,output_file,max_seq_length=512,discard=256):
    with open(output_file,'w') as fout:
        for sentence,label in tqdm(raw_list):
            newlines = reshape_with_cp(sentence,max_seq_length)
            # print(len(newlines))
            for line in newlines:
                # print(line)
                if(len(line) < discard): 
                    # print('discard')
                    continue
                jsonLine = '{"text":"'+' '.join(item for item in line)+'","label":'+str(label)+'}'
                # print(jsonLine)
                fout.write(jsonLine+'\n')

def generate_json_file_for_classification_from_list_Long(raw_list,output_file,max_seq_length=512,discard=256):
    with open(output_file,'w') as fout:
        for sentence,label in tqdm(raw_list):
            newlines = reshape(sentence,max_seq_length)
            # print(len(newlines))
            for line in newlines:
                # print(line)
                if(len(line) < discard): 
                    # print('discard')
                    continue
                if 'ϪϼѾӀ' in line[0:5]:
                    print('skip')
                    continue
                jsonLine = '{"text":"'+' '.join(item for item in line)+'","label":'+str(label)+'}'
                # print(jsonLine)
                fout.write(jsonLine+'\n')

def separate_dataset(input_dir,input_json_file,valid_rate=0.05,test_rate=0.05):
    #name the three file
    train_file=os.path.join(input_dir,input_json_file.split('.')[0]+'_train.json')
    valid_file=os.path.join(input_dir,input_json_file.split('.')[0]+'_valid.json')
    test_file=os.path.join(input_dir,input_json_file.split('.')[0]+'_test.json')
    with open(input_json_file,'r') as fin,open(train_file,'w') as train, \
        open(valid_file,'w') as valid,open(test_file,'w') as test:
        lines = fin.readlines()

        #the index of valid and test
        lengthTotal = len(lines)
        random_idx = random.sample(range(lengthTotal), int(lengthTotal*(valid_rate+test_rate)))
        lengthRandom = len(random_idx)
        #the index of valid part
        valid_idx = random.sample(range(lengthRandom), int(lengthRandom*(valid_rate/(valid_rate+test_rate))))
        index = [0 for _ in range(lengthTotal)]  # 0 for train,1 for valid,2 for test
        for i in random_idx:
            index[i] = 2
        for i in valid_idx:
            index[random_idx[i]] = 1
        for i,line in enumerate(index):
            if index[i] == 0:train.write(lines[i])
            elif index[i] == 1:valid.write(lines[i])
            elif index[i] == 2:test.write(lines[i])
            else:raise(BaseException('no such idx'))

def getTrain_and_Value_DataSet_for_Token(input_dir):
    txts = traverse_words_dir_recurrence(os.path.join(input_dir,'subtxt'))
    labellists = traverse_words_dir_recurrence(os.path.join(input_dir,'label'))
    # print(labels)
    data_list=[]
    for idx,file in tqdm(enumerate(txts)):
        # print(root,h,file)
        # print(file)
        fn = file.split('.')[0]+'.label'
        label_file = open(os.path.join(input_dir,'label',fn),'r')
        labellist=eval(label_file.readline())
        # print(type(labellist))
        # print(labellist)
        # print(label)
        # print(label)
        # print(len(labellist))


        with open(os.path.join(input_dir,'subtxt',file),'r') as f:
            line = f.readline()
            line = line.replace('\n','')
            # print(len(line.split(' ')))
            # exit()
        data_list.append([line,labellist])
    return data_list

def generate_json_file_for_classification_from_list_Long_for_Token(raw_list,output_file,max_seq_length=512,discard=256):
    with open(output_file,'w') as fout:
        for sentence,labels in tqdm(raw_list):
            sentence = sentence.strip()
            newlines,newlabels= reshape_with_uc_and_label(sentence,labels,max_seq_length)
            # print(len(newlines))
            for idx,line in enumerate(newlines):
                if len(newlabels) == 0:
                    break
                # print(line)
                if(len(line) < discard): 
                    # print('discard')
                    continue
                # print(len(line))
                # print(len(newlabels[idx]))
                jsonLine = '{"text":"'+' '.join(item for item in line)+'","label":'+str(newlabels[idx])+'}'
                # print(jsonLine)
                fout.write(jsonLine+'\n')

def reshape_with_uc_and_label(lst,labellst,n):
    # print(len(lst.split(' ')))
    # print(len(labellst))
    lst = lst.split(' ')
    # end = len(lst)//n
    res=[]
    res_label=[]
    tmp=[]
    tmp_label=[]
    isStart = True
    idx= 0
    count = 0  
    for i,item in tqdm(enumerate(lst)):
        if isStart:
            tmp.append(item)
            tmp_label.append(labellst[idx])
            # print(count)
            isStart = False
            if item[-2:] != '@@':
                isStart = True
        elif item[-2:] == '@@':
            tmp.append(item)
            # print('i',i)
            # print('idx',idx)
            tmp_label.append(labellst[idx])
            isStart = False
        else:
            # print(item)
            tmp.append(item)
            tmp_label.append(labellst[idx])
            idx+=1
            isStart = True
        count+=1
        if(count == n):
            count=0
            res.append(tmp)
            res_label.append(tmp_label)
            tmp=[]
            tmp_label=[]
       
    if(count != n):
        res.append(tmp)
        res_label.append(tmp_label)
    # print(idx)
    # print(len([item for li in res_label for item in li ]))
    # print(res_label)
    # exit()
    return res,res_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the cp events file to words.')
    parser.add_argument('--input_path',default='./../../dataSet/MAESTRO/wordsCP', help='words path')
    parser.add_argument('--output_path',default='./../../dataSet/MAESTRO/unicodesCP',help='output path')
    args = parser.parse_args()
    # data_list = getTrain_and_Value_DataSet_for_sequence(args.input_path)
    # val_list,train_list = cut_list(data_list,0.1)
    # generate_json_file_for_classification_from_list_Long(val_list,os.path.join(args.output_path,'valid.json'),max_seq_length=510,discard=0)
    # generate_json_file_for_classification_from_list_Long(train_list,os.path.join(args.output_path,'train.json'),max_seq_length=510,discard=0)

    data_list = getTrain_and_Value_DataSet_for_Token(args.input_path)
    val_list,train_list = cut_list(data_list,0.1)
    generate_json_file_for_classification_from_list_Long_for_Token(val_list,os.path.join(args.output_path,'valid.json'),max_seq_length=510,discard=0)
    generate_json_file_for_classification_from_list_Long_for_Token(train_list,os.path.join(args.output_path,'train.json'),max_seq_length=510,discard=0)

    




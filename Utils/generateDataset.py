import os
import random
from tqdm import tqdm
import pickle
import numpy as np
import re
DEFAULT_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])  

def getTrain_and_Value_DataSet_for_composer(input_dir,dict_file=None):
    data_list = []
    if dict_file is None:
        composer2num={}
        num2composer={}
        index = -1
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=root.split('/')[-1]
                if not(composer2num.get(label) is not None):
                    index=index+1
                    composer2num[label]=index
                    num2composer[index]=label
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence=sentence[0:len(sentence)-1]+'.'
                data_list.append([sentence,composer2num[label]])
        path_dict=os.path.join('.','dictionary.pkl')
        pickle.dump((composer2num, num2composer),open(path_dict, 'wb'))
        return data_list
    else:
        composer2num,num2composer = pickle.load(open(dict_file,'rb'))
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=root.split('/')[-1]
                if not(composer2num.get(label)):
                    raise('error:no '+label+' in this dictionary!')
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence[len(sentence)-1]='.'
                data_list.append([sentence,composer2num[label]])
        return data_list

def getTrain_and_Value_DataSet_for_emo(input_dir,dict_file=None):
    data_list = []
    if dict_file is None:
        Emo2num={}
        num2Emo={}
        index = -1
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=file[0:2]
                # print(label) 
                if not(Emo2num.get(label) is not None):
                    index=index+1
                    Emo2num[label]=index
                    num2Emo[index]=label
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence=sentence[0:len(sentence)-1]+'.'
                data_list.append([sentence,Emo2num[label]])
        path_dict=os.path.join('.','dictionaryEMO.pkl')
        pickle.dump((Emo2num, num2Emo),open(path_dict, 'wb'))
        return data_list
    else:
        Emo2num,num2Emo = pickle.load(open(dict_file,'rb'))
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=root.split('/')[-1]
                if not(Emo2num.get(label)):
                    raise('error:no '+label+' in this dictionary!')
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence[len(sentence)-1]='.'
                data_list.append([sentence,Emo2num[label]])
        return data_list

def getTrain_and_Value_Dataset_for_genre(input_dir,dict_file = None):
    data_list = []
    if dict_file is None:
        genre2num={}
        num2genre={}
        index = -1
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=root.split('/')[-1]
                # print(label)
                # exit()
                # print(label) 
                if not(genre2num.get(label) is not None):
                    index=index+1
                    genre2num[label]=index
                    num2genre[index]=label
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence=sentence[0:len(sentence)-1]+'.'
                data_list.append([sentence,genre2num[label]])
        path_dict=os.path.join('.','dictionarygenre.pkl')
        pickle.dump((genre2num, num2genre),open(path_dict, 'wb'))
        return data_list
    else:
        genre2num,num2genre = pickle.load(open(dict_file,'rb'))
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=root.split('/')[-1]
                if not(genre2num.get(label)):
                    raise('error:no '+label+' in this dictionary!')
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence[len(sentence)-1]='.'
                data_list.append([sentence,genre2num[label]])
        return data_list    

def getTrain_and_Value_Dataset_for_dance(input_dir,dict_file = None):
    data_list = []
    pattern = re.compile(r'[^\d]+')
    if dict_file is None:
        dance2num={}
        num2dance={}
        index = -1
        for root, h, files in os.walk(input_dir):
            for file in files:
                print(root,h,file)
                # label=file.split('/')[-1]
                label = pattern.search(file).group()
                print(label)
                # exit()
                if not(dance2num.get(label) is not None):
                    index=index+1
                    dance2num[label]=index
                    num2dance[index]=label
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence=sentence[0:len(sentence)-1]+'.'
                data_list.append([sentence,dance2num[label]])
        path_dict=os.path.join('.','dictionaryDance.pkl')
        pickle.dump((dance2num, num2dance),open(path_dict, 'wb'))
        return data_list
    else:
        dance2num,num2dance = pickle.load(open(dict_file,'rb'))
        for root, h, files in os.walk(input_dir):
            for file in files:
                # print(root,h,file)
                label=root.split('/')[-1]
                if not(dance2num.get(label)):
                    raise('error:no '+label+' in this dictionary!')
                # print(label)
                sentence=''
                with open(os.path.join(root,file),'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        sentence+=(line+' ')
                sentence[len(sentence)-1]='.'
                data_list.append([sentence,dance2num[label]])
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

def generate_json_file_for_classification_from_list(raw_list,output_file):
    with open(output_file,'w') as fout:
        for sentence,label in raw_list:
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

def getTrain_and_Value_DataSet_for_Velocity(input_dir,Unicode_dict_file):
    data_list = []
    word2unic,unic2word = pickle.load(open(Unicode_dict_file,'rb'))
    for root, h, files in os.walk(input_dir):
        for file in files:
            labels=[]
            sentence=''
            with open(os.path.join(root,file),'r') as f:
                for line in f:
                    tokenList = line.replace('\n','').split(' ')
                    tokens=[]
                    for token in tokenList:
                        # print(token)
                        for unic in token:
                            cp=unic2word[unic]
                            if cp.startswith('Note_Velocity_'):
                                # print('token:',unic)
                                velocity = cp[14:]
                                # print('compound word:',cp)
                                # print('velocity:',velocity)
                                velocity_index = np.searchsorted(DEFAULT_VELOCITY_BINS, velocity, side='right')
                                # print('velocity_index',velocity_index)
                                label = velocity_index
                                tokens.append(token)
                            else:
                                label = 0
                        if label == 0:
                            continue
                        labels.append(label)
                    sentence+=(' '.join(token for token in tokens))
                    # print(labels)
                    # print(sentence)
                    assert len(labels) == len(sentence.split(' '))
                    # exit()
            # sentence[len(sentence)-1]='.'
            data_list.append([sentence,labels])
            # print(data_list)
            # print(len(sentence.split(' ')))
            # print(len(labels))
    return data_list

def generate_json_file_for_classification_from_list_Long_for_velocity(raw_list,output_file,max_seq_length=512,discard=256):
    with open(output_file,'w') as fout:
        for sentence,labels in tqdm(raw_list):
            sentence = sentence.strip()
            newlines,newlabels= reshape_with_cp_and_label(sentence,labels,max_seq_length)
            # print(len(newlines))
            for idx,line in enumerate(newlines):
                if len(newlabels) == 0:
                    break
                # print(line)
                if(len(line) < discard): 
                    # print('discard')
                    continue
                print(len(line))
                print(len(newlabels[idx]))
                jsonLine = '{"text":"'+' '.join(item for item in line)+'","label":'+str(newlabels[idx])+'}'
                # print(jsonLine)
                fout.write(jsonLine+'\n')

if __name__ == '__main__':
    # data_list = getTrain_and_Value_DataSet_for_composer('~/dataSet/joann8512-Pianist8-ab9f541/unicodesCP')
    # val_list,train_list = cut_list(data_list,0.1)
    # generate_json_file_for_classification_from_list_Long(val_list,'~/dataSet/joann8512-Pianist8-ab9f541/val_composer_long_withoutBPE.json',max_seq_length=510,discard=0)
    # generate_json_file_for_classification_from_list_Long(train_list,'~/dataSet/joann8512-Pianist8-ab9f541/train_composer_long_withoutBPE.json',max_seq_length=510,discard=0)

    # data_list = getTrain_and_Value_DataSet_for_emo('~/dataSet/EMOPIA_1.0/unicodesCP')
    # val_list,train_list = cut_list(data_list,0.1)
    # generate_json_file_for_classification_from_list_Long(val_list,'~/dataSet/EMOPIA_1.0/val_emo_lang_withoutBPE.json',max_seq_length=510,discard=128)
    # generate_json_file_for_classification_from_list_Long(train_list,'~/dataSet/EMOPIA_1.0/train_emo_lang_withoutBPE.json',max_seq_length=510,discard=128)
    # with open('train_emo.json','r') as f:
    #     for line in f:
    #         print(len(line.split(' ')))
    # a=[]
    # for i in range(0,18,5):
    #     print(i,i+5,sep=',')
    # generate_json_file('~/dataSet/octuple/mergedTrainOctuple.txt','~/dataSet/octuple/mergedTrainOctuple.json')
    # generate_json_file('~/dataSet/octuple/mergedValidOctuple.txt','~/dataSet/octuple/mergedValidOctuple.json')
    # generate_json_file('~/dataSet/octuple/mergedTestOctuple.txt','~/dataSet/octuple/mergedTestOctuple.json')
    # val_list,train_list = getTrain_and_Value_Dataset_for_genre('~/muzic-main/musicbert/topmagd_data_raw/0')
    # val_list2,train_list2 = getTrain_and_Value_Dataset_for_genre('~/muzic-main/musicbert/topmagd_data_raw/1','~/muzic-main/musicbert/topmagd_data_raw/0/dictionaryGenre.pkl')
    # generate_json_file_for_classification_from_list_Long(val_list,'~/muzic-main/musicbert/topmagd_data_raw/0/val_genre_long.json',max_seq_length=1000,discard=64)
    # generate_json_file_for_classification_from_list_Long(train_list,'~/muzic-main/musicbert/topmagd_data_raw/0/train_genre_long.json',max_seq_length=1000,discard=64)
    # data_list = getTrain_and_Value_DataSet_for_Velocity('~/dataSet/PoP909_Pure/UnicodesCP',
    #                                         '~/dataSet/PianoDataset/unicodesCP/dictionary.pkl')
    # val_list,train_list = cut_list(data_list,0.1)
    # generate_json_file_for_classification_from_list_Long_for_velocity(val_list,'~/dataSet/PoP909_Pure/val_velocity_lang_withoutBPE.json',max_seq_length=510,discard=0)
    # generate_json_file_for_classification_from_list_Long_for_velocity(train_list,'~/dataSet/PoP909_Pure/train_velocity_lang_withoutBPE.json',max_seq_length=510,discard=0)
 
    # data_list = getTrain_and_Value_DataSet_for_composer('~/dataSet/joann8512-Pianist8-ab9f541/unicodesCP')
    # val_list,train_list = cut_list(data_list,0.1)
    # generate_json_file_for_classification_from_list_Long(val_list,'~/dataSet/joann8512-Pianist8-ab9f541/val_composer_long_withoutBPE.json',max_seq_length=510,discard=0)
    # generate_json_file_for_classification_from_list_Long(train_list,'~/dataSet/joann8512-Pianist8-ab9f541/train_composer_long_withoutBPE.json',max_seq_length=510,discard=0)

    data_list = getTrain_and_Value_Dataset_for_genre('~/dataSet/genre/subUnicodesCP')
    val_list,train_list = cut_list(data_list,0.1)
    generate_json_file_for_classification_from_list_Long(val_list,'~/dataSet/genre/val_genre_long_BPE.json',max_seq_length=510,discard=0)
    generate_json_file_for_classification_from_list_Long(train_list,'~/dataSet/genre/train_genre_long_BPE.json',max_seq_length=510,discard=0)



    # data_list = getTrain_and_Value_Dataset_for_dance('~/dataSet/Nottingham/Dataset/UnicodesCP')
    # val_list,train_list = cut_list(data_list,0.1)
    # generate_json_file_for_classification_from_list_Long(val_list,'~/dataSet/Nottingham/Dataset/val_dance_long_NOBPE.json',max_seq_length=510,discard=0)
    # generate_json_file_for_classification_from_list_Long(train_list,'~/dataSet/Nottingham/Dataset/train_dance_long_NOBPE.json',max_seq_length=510,discard=0)


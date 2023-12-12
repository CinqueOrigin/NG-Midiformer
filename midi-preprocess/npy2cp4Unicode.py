import numpy as np
import miditoolkit
import os
import copy
import argparse
import pickle
from tqdm import tqdm
import json
def compare_dicts(d1, d2):
    """
    递归比较两个字典及其值是否相等
    """
    if len(d1) != len(d2):
        return False

    for k, v1 in d1.items():
        if k not in d2:
            return False

        v2 = d2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            # 递归比较嵌套字典
            if not compare_dicts(v1, v2):
                return False
        elif v1 != v2:
            return False

    return True

def num2unic(index):
    indexTemp=hex(index)
    indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
    indexTemp=indexTemp.decode('unicode_escape')
    return indexTemp

def makedict(CP_dict,dict_name=None,baseNum=1000):
    word2unic = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}}
    unic2word = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}}
    f = open(CP_dict,'rb')
    event2word,word2event = pickle.load(f)
    index = baseNum
    for event in word2event.keys():
        for key,value in word2event[event].items():
            unic = num2unic(int(index))
            word2unic[event][key] = unic
            unic2word[event][unic] = key
            index+=1
        # print(len(event2word[event]))
        # print(len(word2event[event]))
        # print(len(word2unic[event]))
        # print(len(unic2word[event]))
    if dict_name is None:
        out_name = 'cp4dict2unicode.pkl'
        fout = open(out_name,'wb')
        pickle.dump((word2unic,unic2word),fout)
    else:
        fout = open(dict_name,'wb')
        pickle.dump((word2unic,unic2word),fout)
    return word2unic,unic2word

def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(".npy"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    return file_list

def npy2cp4Unicode_txt(input_file,mode='pretrain',output='',dict_file = None,cp_dict=None,baseNum = 1000):
    keys = ['Bar', 'Position', 'Pitch', 'Duration']
    npy_list = traverse_words_dir_recurrence(input_file)
    print(npy_list)
    # exit()
    data = []
    for npyfile in npy_list:
        npy = np.load(os.path.join(input_file,npyfile))
        print(npy.shape)
        data.append(np.load(os.path.join(input_file,npyfile)))
    # exit()
    res = np.empty((0,data[0].shape[1],data[0].shape[2]),dtype=np.int64)
    for npy in data:
        res = np.concatenate((res,npy),axis=0)
    data = res
    # print(npy.shape)
    print(data.shape)
    if dict_file is not None:
        word2unic,unic2word = pickle.load(open(dict_file,'rb'))
    else:
        dict_name = os.path.join(os.path.dirname(input_file),'cp4dict2unicode.pkl')
        word2unic,unic2word = makedict(cp_dict,dict_name,baseNum)
    with open(output,'w') as f_out:
        for midi in tqdm(data):
            Words = []
            for cp in midi:
                word=''
                for idx,c in enumerate(cp):
                    word+=word2unic[keys[idx]][c]
                Words.append(word)
            # print(' '.join(word for word in Words))
            f_out.write(' '.join(word for word in Words)+'\n')
            # exit()

def npy2cp4Unicode_token_json(input_file,input_ans_file,mode='pretrain',output='',dict_file = None,cp_dict=None,baseNum = 1000):
    keys = ['Bar', 'Position', 'Pitch', 'Duration']
    # npy_list = traverse_words_dir_recurrence(input_file)
    # print(npy_list)
    # exit()
    data = np.load(input_file)
    ans = np.load(input_ans_file)
    # for npyfile in npy_list:
    #     npy = np.load(os.path.join(input_file,npyfile))
    #     print(npy.shape)
    #     data.append(np.load(os.path.join(input_file,npyfile)))
    # # exit()
    # res = np.empty((0,data[0].shape[1],data[0].shape[2]),dtype=np.int64)
    # for npy in data:
    #     res = np.concatenate((res,npy),axis=0)
    # data = res
    # # print(npy.shape)
    # print(data.shape)
    if dict_file is not None:
        word2unic,unic2word = pickle.load(open(dict_file,'rb'))
    else:
        dict_name = os.path.join(os.path.dirname(input_file),'cp4dict2unicode.pkl')
        word2unic,unic2word = makedict(cp_dict,dict_name,baseNum)
    with open(output,'w') as f_out:
        for i,midi in tqdm(enumerate(data)):
            # print(midi.shape)
            # print(ans[i].shape)
            Words = []
            labels=[]
            for cp in midi:
                word=''
                for idx,c in enumerate(cp):
                    word+=word2unic[keys[idx]][c]
                label = ans[i]
                Words.append(word)
                labels.append(label)
            # print(' '.join(word for word in Words))
            # labelstr = '['+','.join(label for label in labels[idx])+']'
            jsonLine = '{"text":"'+' '.join(item for item in Words)+'","label":'+str(labels[idx].tolist())+'}'
            # print(jsonLine)
            f_out.write(jsonLine+'\n')
            # exit()

def npy2cp4Unicode_seq_json(input_file,input_ans_file,mode='pretrain',output='',dict_file = None,cp_dict=None,baseNum = 1000):
    keys = ['Bar', 'Position', 'Pitch', 'Duration']
    # npy_list = traverse_words_dir_recurrence(input_file)
    # print(npy_list)
    # exit()
    data = np.load(input_file)
    ans = np.load(input_ans_file)
    # for npyfile in npy_list:
    #     npy = np.load(os.path.join(input_file,npyfile))
    #     print(npy.shape)
    #     data.append(np.load(os.path.join(input_file,npyfile)))
    # # exit()
    # res = np.empty((0,data[0].shape[1],data[0].shape[2]),dtype=np.int64)
    # for npy in data:
    #     res = np.concatenate((res,npy),axis=0)
    # data = res
    # # print(npy.shape)
    # print(data.shape)
    if dict_file is not None:
        word2unic,unic2word = pickle.load(open(dict_file,'rb'))
    else:
        dict_name = os.path.join(os.path.dirname(input_file),'cp4dict2unicode.pkl')
        word2unic,unic2word = makedict(cp_dict,dict_name,baseNum)
    with open(output,'w') as f_out:
        for i,midi in tqdm(enumerate(data)):
            # print(midi.shape)
            # print(ans[i].shape)
            Words = []
            labels=[]
            for cp in midi:
                word=''
                for idx,c in enumerate(cp):
                    word+=word2unic[keys[idx]][c]
                label = ans[i]
                Words.append(word)
                labels.append(label)
            # print(' '.join(word for word in Words))
            jsonLine = '{"text":"'+' '.join(item for item in Words)+'","label":'+str(labels[idx])+'}'
            # print(jsonLine)
            f_out.write(jsonLine+'\n')
            # exit()

def convert_npy_2_txt(input_file,input_file2,output_dir,dict_file):
    count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    keys = ['Bar', 'Position', 'Pitch', 'Duration']
    # word2unic,unic2word = pickle.load(open(dict_file,'rb'))
    with open(input_file,'r') as f_in:
        for i,midi in tqdm(enumerate(f_in)):
            data = json.loads(midi)
            text = data['text']
            label = data['label']
            # print(text)
            # print(label)
            # exit()
            fn = str(count)+'.txt'
            label_fn = str(count)+'.label'
            if not os.path.exists(os.path.join(output_dir,'txt')):
                 os.makedirs(os.path.join(output_dir,'txt'))
            if not os.path.exists(os.path.join(output_dir,'label')):
                 os.makedirs(os.path.join(output_dir,'label'))
            with open(os.path.join(output_dir,'txt',fn),'w') as fout:
                fout.write(text+'\n')
            with open(os.path.join(output_dir,'label',label_fn),'w') as f_label:
                f_label.write(str(label)+'\n')
            count+=1
    with open(input_file2,'r') as f_in:
        for i,midi in tqdm(enumerate(f_in)):
            data = json.loads(midi)
            text = data['text']
            label = data['label']
            # print(text)
            # print(label)
            # exit()
            fn = str(count)+'.txt'
            label_fn = str(count)+'.label'
            if not os.path.exists(os.path.join(output_dir,'txt')):
                 os.makedirs(os.path.join(output_dir,'txt'))
            if not os.path.exists(os.path.join(output_dir,'label')):
                 os.makedirs(os.path.join(output_dir,'label'))
            with open(os.path.join(output_dir,'txt',fn),'w') as fout:
                fout.write(text+'\n')
            with open(os.path.join(output_dir,'label',label_fn),'w') as f_label:
                f_label.write(str(label)+'\n')
            count+=1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the cp events file to words.')
    parser.add_argument('--input_path',default='./../../dataSet/MAESTRO/wordsCP', help='words path')
    parser.add_argument('--input_ans',default='./../../dataSet/MAESTRO/wordsCP', help='words path')
    parser.add_argument('--output_path',default='./../../dataSet/MAESTRO/unicodesCP',help='output path')
    parser.add_argument('--baseNum', default=1000, help='the start of Unicode encoding')
    parser.add_argument('--dict_file', default=None, help='dict file of CP')
    parser.add_argument('--CP_dict', default=None, help='dict file of CP')
    parser.add_argument('--mode', default='pretrain', help='dict file of CP')
    args = parser.parse_args()
    # makedict('~/MIDI-BERT/data_creation/prepare_data/dict/CP.pkl')
    if args.mode == 'pretrain':
        npy2cp4Unicode_txt(args.input_path,args.mode,args.output_path,
                   args.dict_file,args.CP_dict,args.baseNum)
    elif args.mode == 'token':
        npy2cp4Unicode_token_json(args.input_path,args.input_ans,args.mode,args.output_path,
                   args.dict_file,args.CP_dict,args.baseNum)
    elif args.mode == 'seq':
        npy2cp4Unicode_seq_json(args.input_path,args.input_ans,args.mode,args.output_path,
                   args.dict_file,args.CP_dict,args.baseNum)
    else:
        convert_npy_2_txt(args.input_path,args.input_ans,args.output_path,
                   args.dict_file)
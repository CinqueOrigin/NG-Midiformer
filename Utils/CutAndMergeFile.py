import os 
from tqdm import tqdm

def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(".txt"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    return file_list

def cut_file_to_list(input_file,output_dir,num=100):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(input_file,'r') as fin:
        index = 0
        for line in tqdm(fin):
            fn = os.path.basename(input_file)[:-4]
            out_path = os.path.join(output_dir,fn+'_'+str(index)+'.txt')
            with open(out_path,'a') as fout:
                fout.write(line)
                fout.close()
            index+=1
            if index>=num:
                index = 0

def mergeFile(input_dir,output_file,num=100):
    file_list = traverse_words_dir_recurrence(input_dir)
    print(file_list)
    with open (output_file,'w') as fout:
        for file in tqdm(file_list):
            with open(os.path.join(input_dir,file)) as fin:
                fout.write(fin.read())
                fout.write('\n')

def mergeGenreFile(input_dir,output_file,num=100):
    file_list = traverse_words_dir_recurrence(input_dir)
    data_list = []
    for file in file_list:
        if 'test' in file:data_list.append(file)
    # print(data_list)
    # exit()
    with open (output_file,'w') as fout:
        for file in tqdm(data_list):
            with open(os.path.join(input_dir,file)) as fin:
                fout.write(fin.read())
                fout.write('\n')


if __name__ == '__main__':
    # cut_file_to_list('~/workSpace/musicbert/output/midi_train_unicode.txt',
    # '~/workSpace/musicbert/output/subTrain',num=7)

    # mergeGenreFile('~/muzic-main/musicbert/topmagd_data_raw',
    # '~/muzic-main/musicbert/topmagd_data_raw/test.label')

    mergeFile('~/workSpace/musicbert/output/subTrainBped',
    '~/workSpace/musicbert/output/subTrainBped/mergedTrainOctuple.txt')
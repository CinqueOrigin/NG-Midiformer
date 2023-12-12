from tqdm import tqdm
import os


def convert_octuple_genre_to_json(input_txt_file,input_label_file,output_file):
    counts={}
    with open(output_file,'w') as fout:
        with open(input_txt_file,'r') as f_txt:
            with open(input_label_file,'r') as f_label:
                txts = f_txt.readlines()
                labels = f_label.readlines()
                length = len(txts)
                for i in tqdm(range(length)):
                    label_temps=labels[i]
                    labelss=[]
                    for label in label_temps.split(' '):
                        label = label.replace('\n','')
                        if label in counts:
                            counts[label] +=1
                        else:
                            counts[label] = 1
                        labelss.append(label)
                    label_str=','.join(labelss)
                    print(label_str)
                    # exit()
                    fout.write('{"text":"'+txts[i].replace('\n','')+'","label":"'+label_str+'"}\n')
                print(len(counts))
                        # print('"{text:"'+txts[i].replace('\n','')+'"，label:"'+label.replace('\n','').split('_')[0]+'"}\n')

def convert_octuple_style_to_json(input_txt_file,input_label_file,output_file):
    counts={}
    with open(output_file,'w') as fout:
        with open(input_txt_file,'r') as f_txt:
            with open(input_label_file,'r') as f_label:
                txts = f_txt.readlines()
                labels = f_label.readlines()
                length = len(txts)
                for i in tqdm(range(length)):
                    label_temps=labels[i]
                    labelss=[]
                    for label in label_temps.split(' '):
                        label = label.replace('\n','')
                        if label in counts:
                            counts[label] +=1
                        else:
                            counts[label] = 1
                        labelss.append(label)
                    label_str=','.join(labelss)
                    print(label_str)
                    # exit()
                    fout.write('{"text":"'+txts[i].replace('\n','')+'","label":"'+label_str+'"}\n')
                print(len(counts))
                        # print('"{text:"'+txts[i].replace('\n','')+'"，label:"'+label.replace('\n','').split('_')[0]+'"}\n')

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

def convert_melGenre_to_json(input_file,label_file,output_file):
 with open(input_file, 'r') as f_in,open(label_file, 'r') as f_label:
     lines = f_in.readlines()
     labels = f_label.readlines()
     with open(output_file, 'w') as f_out:
        for idx,line in enumerate(lines):
            f_out.write('{"text":"'+line.replace('\n','')+'","label":"'+labels[idx].replace('\n','')+'"}\n')
         


if __name__ == '__main__':
    # convert_octuple_genre_to_json('~/workSpace/musicbert/topmagd_data_raw/0/train_unicode_bped.txt',
    #                               '~/workSpace/musicbert/topmagd_data_raw/0/train.label',
    #                               '~/workSpace/musicbert/topmagd_data_raw/0/train.json')
    convert_melGenre_to_json('~/dataSet/genre/UnicodesCP/mergeUnicodeFilePiano.txt',
                             '~/dataSet/genre/UnicodesCP/label.txt',
                             '~/dataSet/genre/UnicodesCP/mergeUnicodeFilePianoGenreMel.json')
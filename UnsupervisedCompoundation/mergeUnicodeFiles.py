import os
from tqdm import tqdm


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

def mergeUnicodeFiles(input_path,output_path):
    unic_files = traverse_words_dir_recurrence(input_path)

    path_outfile = os.path.join(output_path, 'mergedUnicodes.txt')
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    with open(os.path.join(output_path,'mergedUnicodes.txt'),'w') as fout:
        for file in tqdm(unic_files):
            with open(os.path.join(input_path,file),'r') as fin:
                for line in fin:
                    line =line.replace('\n',' ')
                    fout.write(line)
                fout.write('.\n')

    
if __name__ == '__main__':
    
    mergeUnicodeFiles('./../../dataSet/MAESTRO/unicodesCP','./../../dataSet/MAESTRO')

    


import os
import fastBPE
from tqdm import tqdm
from subprocess import run
import argparse
import multiprocessing

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

def run_cmd(cmd_str=''):
    # cmd_str = 'echo 2'
    run(cmd_str,shell=True)

def applyFastBPE(words_dir,output_dir,codes_dir):
    words_files =traverse_words_dir_recurrence(words_dir)

    os.makedirs(output_dir, exist_ok=True)
    cmd_Str_list=[]
    with multiprocessing.Pool(processes=1) as pool:
        for file in tqdm(words_files):
            output_file=os.path.join(output_dir,file)

            fn = os.path.basename(output_file)
            os.makedirs(output_file[:-len(fn)], exist_ok=True)

            input_file=os.path.join(words_dir,file)
            cmd_str='./fastBPE/fast applybpe '+output_file+' '+input_file+' '+codes_dir
            cmd_Str_list.append(cmd_str)
        pool.map(run_cmd,cmd_Str_list)
        pool.close()
        pool.join()
            # run_cmd(cmd_str)

def applyFastBPEForOneFile(words_path,output_dir,codes_dir):
    # words_files =traverse_words_dir_recurrence(words_dir)
    words_file = words_path

    # os.makedirs(output_dir, exist_ok=True)
    # fn = os.path.basename(words_path)
    # output_file=os.path.join(output_dir,fn)
    output_file = output_dir

    cmd_str='./fastBPE/fast applybpe '+output_file+' '+words_path+' '+codes_dir
    run_cmd(cmd_str)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='apply fastBPE')
    parser.add_argument('--txt_file',default='~/workSpace/musicbert/output/midi_test_unicode.txt', help='words path')
    parser.add_argument('--output_dir',default='~/workSpace/musicbert/output/midi_test_unicode_bped.txt',help='output path')
    parser.add_argument('--codes_file',default='~/dataSet/lmd/UnicodesCP/codes_1000_Piano.txt',help='output path')
    args = parser.parse_args()
    # applyFastBPEForOneFile(args.txt_file,
    #                       args.output_dir,
    #                        args.codes_file)
    # os.system('mkdir -p {}'.format(args.output_dir))
    applyFastBPE(args.txt_file,args.output_dir,args.codes_file)
    # applyFastBPE('~/dataSet/genre/UnicodesCP','~/dataSet/genre/subUnicodesCP','~/dataSet/PianoDataset/unicodesCP/codes_1000_Piano')


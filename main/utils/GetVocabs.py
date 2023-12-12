import os
from tqdm import tqdm
import math
pos_resolution = 16  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1000  # window length max
sample_overlap_rate = 1
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

def getVocab(input_file,output_file = None):
    # create vocab.txt from input file without superving
    if output_file is None:
        output_file = 'vocab.txt' 
    with open(output_file,'w') as fout:
        set1=set()
        with open(input_file,'r') as f:
            for line in tqdm(f):
                line=line.replace('\n','')
                line = line.split(' ')
                for word in line:
                    if word not in set1:
                        set1.add(word)
                        fout.write(word+'\n')

    print(len(set1))

def v2e(x):
    return x // velocity_quant

def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e

def get_ts_list():
    ts_dict={}
    ts_list = list()
    for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
        for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
            ts_dict[(j, 2 ** i)] = len(ts_dict)
            ts_list.append((j, 2 ** i))
    return ts_list 

def convert_index_to_unicode(index):
    indexTemp=hex(index)
    indexTemp=(b"\u"+str(indexTemp).encode()).replace(b'x',b'')
    indexTemp=indexTemp.decode('unicode_escape')
    return indexTemp

def get_dictionary(file_name,baseNum=1000):
    #get dictionary file from octuple to unicode
    num = baseNum
    ts_list = get_ts_list()
    with open(file_name, 'w') as f:
        for j in range(bar_max):
            unic=convert_index_to_unicode(num)
            print('<0-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            unic=convert_index_to_unicode(num)
            print('<1-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(max_inst + 1 + 1):
            # max_inst + 1 for percussion
            unic=convert_index_to_unicode(num)
            print('<2-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(2 * max_pitch + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            unic=convert_index_to_unicode(num)
            print('<3-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(duration_max * pos_resolution):
            unic=convert_index_to_unicode(num)
            print('<4-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(v2e(max_velocity) + 1):
            unic=convert_index_to_unicode(num)
            print('<5-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(len(ts_list)):
            unic=convert_index_to_unicode(num)
            print('<6-{}>'.format(j), unic, file=f)
            num+=1
        for j in range(b2e(max_tempo) + 1):
            unic=convert_index_to_unicode(num)
            print('<7-{}>'.format(j), unic, file=f)
            num+=1
        unic=convert_index_to_unicode(num)
        print('<s>', unic, file=f)
        num+=1
        unic=convert_index_to_unicode(num)
        print('</s>', unic, file=f)

def get_dictionary_for_vocab(vocab_file,baseNum=1000):
    #get dictionary file from octuple to unicode
    num = baseNum
    ts_list = get_ts_list()
    dic = dict()
    with open(vocab_file, 'w') as f:
        for j in range(bar_max):
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(max_inst + 1 + 1):
            # max_inst + 1 for percussion
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(2 * max_pitch + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(duration_max * pos_resolution):
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(v2e(max_velocity) + 1):
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(len(ts_list)):
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        for j in range(b2e(max_tempo) + 1):
            unic=convert_index_to_unicode(num)
            print(unic, file=f)
            num+=1
        unic=convert_index_to_unicode(num)
        print(unic, file=f)
        num+=1
        unic=convert_index_to_unicode(num)
        print(unic, file=f)

if __name__ == '__main__':
    getVocab('~/dataSet/PianoDataset/cp4/uc4/mergeUnicodeFilePianoBped.txt',
             '~/dataSet/PianoDataset/cp4/uc4/vocab.txt')
    # get_dictionary('~/workSpace/musicbert/lmd_full_data_raw/vocabdict.txt')
    # get_dictionary_for_vocab('~/workSpace/musicbert/lmd_full_data_raw/vocab.txt')
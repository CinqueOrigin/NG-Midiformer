from tqdm import tqdm
import os
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

def Count_for_labels(input_file):
    counts={}
    with open(input_file,'r') as fin:
        for line in fin:
            line = line.strip()
            for label in line.split(' '):
                if label in counts:
                    counts[label] +=1
                else:
                    counts[label] = 1
    print(len(counts))
    for label,count in counts.items():
        print(label,count)

def Count_for_items(input_file,line):
    count = []
    with open(input_file,'r') as fin:
        for idx,line in tqdm(fin):
            line = line.strip()
            items = line.split(' ')
            count.append(len(items))
            
    print(count)
    max_count = count.max()
    min_count = count.min()
    print(max_count,min_count)

def Count_average_items(input_file):
    with open(input_file) as fin:
        print(input_file)
        total = 0
        count_list = []
        for line in tqdm(fin):
            line = line.split(' ')
            count_list.append(len(line))
            total += len(line)
            # if len(count_list) == 1000 :break
            
        print('average:'+str(total/len(count_list)))

def count_duration(input_dir):
    file_list = traverse_words_dir_recurrence(input_dir,input_suffix='.midi')
    import mido
    duration = 0
    count = 0
    for file in file_list:
        midi = mido.MidiFile(os.path.join(input_dir,file))
        duration+=midi.length
        count+=1
    print('total duration of the midi files:',duration/3600,'hours')
    print('average duration of the midi files:',duration/count/3600,'hours')


if __name__ == '__main__':
    # Count_for_labels('~/workSpace/musicbert/topmagd_data_raw/4/test.label')
    # Count_average_items('~/dataSet/PianoDataset/subUnicodesCP/mergeUnicodeFilePiano.txt')
    count_duration('~/dataSet/PianoDataset/midi_analyzed/adl-piano-midi')



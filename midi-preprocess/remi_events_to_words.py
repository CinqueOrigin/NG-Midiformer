import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np

def traverse_events_dir(events_dir):
    """
    从events文件夹中获取所有的events文件名称
    :param events_dir:
    :return:
    """
    file_names = []
    for f in os.listdir(events_dir):
        if f.endswith(".pkl"):
            file_names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    return file_names

def traverse_events_dir_recurrence(events_dir):
    file_list = []
    for root, h, files in os.walk(events_dir):
        for file in files:
            if file.endswith(".pkl"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(events_dir)+1:]
                ext = pure_path.split('.')[-1]
                pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
    return file_list


def main():
    parser = argparse.ArgumentParser(description='Process the remi events file to words.')
    parser.add_argument('--events_path', help='corpus path')
    parser.add_argument('--words_path', help='output path')
    parser.add_argument('--format', default="txt", help='output format')
    args = parser.parse_args()
    # events_files = traverse_events_dir(args.events_path)
    events_files = traverse_events_dir_recurrence(args.events_path)

    if args.format == "bin":
        all_events = []
        for file in events_files:
            for event in pickle.load(open(os.path.join(args.events_path, file + ".pkl"), 'rb')):
                all_events.append('{}_{}'.format(event['name'], event['value']))

        # build
        unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
        event2word = {key: i for i, key in enumerate(unique_events)}
        word2event = {i: key for i, key in enumerate(unique_events)}
        print(' > num classes:', len(word2event))

        # save
        pickle.dump((event2word, word2event), open(os.path.join(args.words_path, "dictionary.pkl"), 'wb'))

        # --- converts to word --- #
        event2word, word2event = pickle.load(open(os.path.join(args.words_path, "dictionary.pkl"), 'rb'))
        for file in tqdm(events_files):
            # events to words
            path_infile = os.path.join(args.events_path, file+".pkl")
            events = pickle.load(open(path_infile, 'rb'))
            words = []
            for event in events:
                word = event2word['{}_{}'.format(event['name'], event['value'])]
                words.append(word)

            # save
            path_outfile = os.path.join(args.words_path, file + '.npy')
            fn = os.path.basename(path_outfile)
            os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
            np.save(path_outfile, words)

    else:
        for file in tqdm(events_files):
            # events to words
            path_infile = os.path.join(args.events_path, file + ".pkl")
            events = pickle.load(open(path_infile, 'rb'))
            words = []
            for event in events:
                word = '{}_{}'.format(event['name'], event['value'])
                words.append(word)

            # save
            path_outfile = os.path.join(args.words_path, file + '.txt')
            fn = os.path.basename(path_outfile)
            os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

            with open(path_outfile, "w") as fout:
                fout.write(" ".join(words)+"\n")


if __name__ == "__main__":
    main()
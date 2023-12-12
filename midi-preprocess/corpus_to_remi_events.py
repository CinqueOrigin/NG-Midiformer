import os
import pickle
import argparse
from tqdm import tqdm

def traverse_corpus_dir(corpus_dir):
    """
    从corpus文件夹中获取所有的corpus文件名称
    :param corpus_dir:
    :return:
    """
    file_names = []
    for f in os.listdir(corpus_dir):
        if f.endswith(".pkl"):
            file_names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    return file_names


def traverse_corpus_dir_recurrence(corpus_dir):
    file_list = []
    for root, h, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(".pkl"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(corpus_dir)+1:]
                ext = pure_path.split('.')[-1]
                pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
    return file_list

# config
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


# define event
def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event


# core functions
def corpus2event_remi_v2(path_infile, path_outfile):
    """
    <<< REMI v2 >>>
    task: 2 track
        1: piano      (note + tempo + chord)
    ---
    remove duplicate position tokens
    """
    data = pickle.load(open(path_infile, 'rb'))

    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_event('Bar', None))

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_events = []

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing] # piano track

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                pos_events.append(create_event('Chord', root+'_'+quality))

            # tempo
            if len(t_tempos):
                pos_events.append(create_event('Tempo', t_tempos[0].tempo))

            # note
            if len(t_notes):
                for note in t_notes:
                    pos_events.extend([
                        create_event('Note_Pitch', note.pitch),
                        create_event('Note_Velocity', note.velocity),
                        create_event('Note_Duration', note.duration),
                    ])

            # collect & beat
            if len(pos_events):
                final_sequence.append(
                    create_event('Beat', (timing-bar_step)//TICK_RESOL))
                final_sequence.extend(pos_events)

    # BAR ending
    final_sequence.append(create_event('Bar', None))

    # EOS
    final_sequence.append(create_event('EOS', None))

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle.dump(final_sequence, open(path_outfile, 'wb'))

    # print(final_sequence)


def main():
    parser = argparse.ArgumentParser(description='Process the corpus file to remi events.')
    parser.add_argument('--corpus_path', help='corpus path')
    parser.add_argument('--output_path', help='output path')
    args = parser.parse_args()

    # corpus_file_names = traverse_corpus_dir(args.corpus_path)
    corpus_file_names = traverse_corpus_dir_recurrence(args.corpus_path)

    for fn in tqdm(corpus_file_names):
        corpus2event_remi_v2(os.path.join(args.corpus_path, fn+".pkl"), os.path.join(args.output_path, fn+".pkl"))

def debug_main():
    corpus2event_remi_v2("D:\\Research\\music-ai\\chinese_song_corpus\\omso-埃玛(草东没有派对).midi.pkl", "D:\\Research\\music-ai\\debug\\omso-埃玛(草东没有派对).pkl")

if __name__ == "__main__":
    # debug_main()

    main()
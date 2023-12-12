import os
import pickle
import argparse
from tqdm import tqdm
import multiprocessing as mp
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

# ---- define event ---- #
''' 8 kinds:
     tempo: 0:   IGN     
            1:   no change
            int: tempo
     chord: 0:   IGN
            1:   no change
            str: chord types
  bar-beat: 0:   IGN     
            int: beat position (1...16)
            int: bar (bar)
      type: 0:   eos    
            1:   metrical
            2:   note
  duration: 0:   IGN
            int: length
     pitch: 0:   IGN
            int: pitch
  velocity: 0:   IGN    
            int: velocity
'''

# event template
compound_event = {
    'tempo': 0,
    'chord': 0,
    'bar-beat': 0,
    'type': 0,
    'pitch': 0,
    'duration': 0,
    'velocity': 0,
}

def create_bar_event():
    meter_event = compound_event.copy()
    meter_event['bar-beat'] = 'Bar'
    meter_event['type'] = 'Metrical'
    return meter_event


def create_piano_metrical_event(tempo, chord, pos):
    meter_event = compound_event.copy()
    meter_event['tempo'] = tempo
    meter_event['chord'] = chord
    meter_event['bar-beat'] = pos
    meter_event['type'] = 'Metrical'
    return meter_event


def create_piano_note_event(pitch, duration, velocity):
    note_event = compound_event.copy()
    note_event['pitch'] = pitch
    note_event['duration'] = duration
    note_event['velocity'] = velocity
    note_event['type'] = 'Note'
    return note_event


def create_eos_event():
    eos_event = compound_event.copy()
    eos_event['type'] = 'EOS'
    return eos_event


# core functions
def corpus2event_cp(path_infile, path_outfile):
    '''
    task: 2 track
        1: piano      (note + tempo)
    ---
    remove duplicate position tokens
    '''
    try:
        data = pickle.load(open(path_infile, 'rb'))
    except:
        raise BaseException('error: file '+path_infile)
        exit()
    print(data['metadata']['global_bpm'],file=open('a.txt','a'))
    return 
    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_bar_event())

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_on = False
            pos_events = []
            pos_text = 'Beat_' + str((timing - bar_step) // TICK_RESOL)

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing]  # piano track

            # metrical
            if len(t_tempos) or len(t_chords):
                # chord
                if len(t_chords):

                    root, quality, bass = t_chords[-1].text.split('_')
                    chord_text = root + '_' + quality
                else:
                    chord_text = 'CONTI'

                # tempo
                if len(t_tempos):
                    tempo_text = 'Tempo_' + str(t_tempos[-1].tempo)
                else:
                    tempo_text = 'CONTI'

                # create
                pos_events.append(
                    create_piano_metrical_event(
                        tempo_text, chord_text, pos_text))
                pos_on = True

            # note
            if len(t_notes):
                if not pos_on:
                    pos_events.append(
                        create_piano_metrical_event(
                            'CONTI', 'CONTI', pos_text))

                for note in t_notes:
                    note_pitch_text = 'Note_Pitch_' + str(note.pitch)
                    note_duration_text = 'Note_Duration_' + str(note.duration)
                    note_velocity_text = 'Note_Velocity_' + str(note.velocity)

                    pos_events.append(
                        create_piano_note_event(
                            note_pitch_text,
                            note_duration_text,
                            note_velocity_text))

            # collect & beat
            if len(pos_events):
                final_sequence.extend(pos_events)

    # BAR ending
    final_sequence.append(create_bar_event())

    # EOS
    final_sequence.append(create_eos_event())

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle.dump(final_sequence, open(path_outfile, 'wb'))

    print('down')


def main():
    parser = argparse.ArgumentParser(description='Process the corpus file to cp events.')
    parser.add_argument('--corpus_path', help='corpus path')
    parser.add_argument('--output_path', help='output path')
    args = parser.parse_args()

    # corpus_file_names = traverse_corpus_dir(args.corpus_path)
    corpus_file_names = traverse_corpus_dir_recurrence(args.corpus_path)
    data=[]
    for fn in tqdm(corpus_file_names):
        path = os.path.join(args.output_path,fn+'.pkl')
        # print(path)
        # if(os.path.exists(path)):
        #     # print('skip')
        #     continue
        data.append([os.path.join(args.corpus_path, fn+".pkl"),os.path.join(args.output_path, fn+".pkl")])
        # corpus2event_cp(os.path.join(args.corpus_path, fn+".pkl"), os.path.join(args.output_path, fn+".pkl"))
    pool = mp.Pool()
    pool.starmap(corpus2event_cp, data)

def debug_main():
    corpus2event_cp("~/dataSet/PoP909_Pure/corpus/001.pkl", "~/dataSet/PoP909_Pure/eventsCP/001.pkl")

if __name__ == "__main__":
    # debug_main()

    main()
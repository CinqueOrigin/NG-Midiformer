import os
import glob
import copy
import librosa
import argparse
import numpy as np
import collections
import pickle
import multiprocessing as mp

from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor

import miditoolkit
from miditoolkit.midi import parser
from miditoolkit.midi.containers import TimeSignature, TempoChange, Marker, Instrument
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser

from chorder import Dechorder


def traverse_midi_dir(midi_dir, input_midi_suffix='midi'):
    """
    从midi文件夹中获取所有的midi文件名称
    :param midi_dir:
    :return:
    """
    #print(midi_dir, input_midi_suffix)
    file_names = []
    for f in os.listdir(midi_dir):
        #print(f)
        if f.endswith("."+input_midi_suffix):
            file_names.append('.'.join(os.path.basename(f).split('.')[:-1]))
    return file_names

def traverse_midi_dir_recurrence(midi_dir,input_midi_suffix='midi'):
    file_list = []
    for root, h, files in os.walk(midi_dir):
        for file in files:
            if file.endswith("."+input_midi_suffix) or file.endswith(".midi"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(midi_dir)+1:]
                ext = pure_path.split('.')[-1]
                pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
    return file_list


# synchronizer

def get_instruments_abs_timing(instruments, tick_to_time):
    return convert_instruments_timing_from_sym_to_abs(instruments, tick_to_time)


def convert_instruments_timing_from_sym_to_abs(instruments, tick_to_time):
    proc_instrs = copy.deepcopy(instruments)
    for instr in proc_instrs:
        for note in instr.notes:
            note.start = float(tick_to_time[note.start])
            note.end = float(tick_to_time[note.end])
    return proc_instrs


def convert_instruments_timing_from_abs_to_sym(instruments, time_to_tick):
    proc_instrs = copy.deepcopy(instruments)
    for instr in proc_instrs:
        for note in instr.notes:
            # find nearest
            note.start = find_nearest_np(time_to_tick, note.start)
            note.end = find_nearest_np(time_to_tick, note.end)
    return proc_instrs


def find_nearest_np(array, value):
    return (np.abs(array - value)).argmin()


def find_first_downbeat(proc_res):
    rythm = np.where(proc_res[:, 1] == 1)[0]
    pos = proc_res[rythm[0], 0]
    return pos


def interp_linear(src, target, num, tail=False):
    src = float(src)
    target = float(target)
    step = (target - src) / float(num)
    middles = [src + step * i for i in range(1, num)]
    res = [src] + middles
    if tail:
        res += [target]
    return res


def estimate_beat(path_audio):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(path_audio)
    proc_res = proc(act)
    return proc_res


def export_audio_with_click(proc_res, path_audio, path_output, sr=44100):
    # extract time
    times_beat = proc_res[np.where(proc_res[:, 1] != 1)][:, 0]
    times_downbeat = proc_res[np.where(proc_res[:, 1] == 1)][:, 0]

    # load
    y, _ = librosa.core.load(path_audio, sr=sr)

    # click audio
    y_beat = librosa.clicks(times=times_beat, sr=sr, click_freq=1200, click_duration=0.5) * 0.6
    y_downbeat = librosa.clicks(times=times_downbeat, sr=sr, click_freq=600, click_duration=0.5)

    # merge
    max_len = max(len(y), len(y_beat), len(y_downbeat))
    y_integrate = np.zeros(max_len)
    y_integrate[:len(y_beat)] += y_beat
    y_integrate[:len(y_downbeat)] += y_downbeat
    y_integrate[:len(y)] += y

    librosa.output.write_wav(path_output, y_integrate, sr)


def align_midi(proc_res, path_midi_input, path_midi_output, ticks_per_beat=480):
    midi_data = parser.MidiFile(path_midi_input)

    # compute tempo
    beats = np.array([0.0] + list(proc_res[:, 0]))
    intervals = np.diff(beats)
    bpms = 60 / intervals
    tempo_info = list(zip(beats[:-1], bpms))

    # get absolute timing of instruments
    tick_to_time = midi_data.get_tick_to_time_mapping()
    abs_instr = get_instruments_abs_timing(midi_data.instruments, tick_to_time)

    # get end time of file
    end_time = midi_data.get_tick_to_time_mapping()[-1]

    # compute time to tick mapping
    resample_timing = []
    for i in range(len(beats) - 1):
        start_beat = beats[i]
        end_beat = beats[i + 1]
        resample_timing += interp_linear(start_beat, end_beat, ticks_per_beat)

    # fill the empty in the tail (using last tick interval)
    last_tick_interval = resample_timing[-1] - resample_timing[-2]
    cur_time = resample_timing[-1]
    while cur_time < end_time:
        cur_time += last_tick_interval
        resample_timing.append(cur_time)
    resample_timing = np.array(resample_timing)

    # new a midifile obj
    midi_res = parser.MidiFile()

    # convert abs to sym
    sym_instr = convert_instruments_timing_from_abs_to_sym(abs_instr, resample_timing)

    # time signature
    first_db_sec = find_first_downbeat(proc_res)
    first_db_tick = find_nearest_np(resample_timing, first_db_sec)
    time_signature_changes = [TimeSignature(numerator=4, denominator=4, time=int(first_db_tick))]

    # tempo
    tempo_changes = []
    for pos, bpm in tempo_info:
        pos_tick = find_nearest_np(resample_timing, pos)
        tempo_changes.append(TempoChange(tempo=float(bpm), time=int(pos_tick)))

    # shift (pickup at the beginning)
    shift_align = ticks_per_beat * 4 - first_db_tick

    # apply shift to tempo
    for msg in tempo_changes:
        msg.time += shift_align

    # apply shift to notes
    for instr in sym_instr:
        for note in instr.notes:
            note.start += shift_align
            note.end += shift_align

    # set attributes
    midi_res.ticks_per_beat = ticks_per_beat
    midi_res.tempo_changes = tempo_changes
    midi_res.time_signature_changes = time_signature_changes
    midi_res.instruments = sym_instr

    # saving
    midi_res.dump(filename=path_midi_output)


def synchronize_one(path_midi_input, path_audio_input, path_midi_output, path_audio_output=None):
    if not os.path.exists(path_midi_input) or not os.path.exists(path_audio_input):
        return
    if os.path.exists(path_midi_output):
        return

    # print("synchronize_one:", path_midi_input)

    # beat tracking
    proc_res = estimate_beat(path_audio_input)
    # export audio with click
    if path_audio_output is not None:
        export_audio_with_click(proc_res, path_audio_input, path_audio_output)
    # export midi file
    align_midi(proc_res, path_midi_input, path_midi_output)


# analyzer

num2pitch = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}


def analyze_one(path_infile, path_outfile):
    if not os.path.exists(path_infile):
        return

    if os.path.exists(path_outfile):
        return

    # print("analyze_one:", path_infile)

    # load
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)
    midi_obj_out = copy.deepcopy(midi_obj)
    notes = midi_obj.instruments[0].notes
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # --- chord --- #
    # exctract chord
    chords = Dechorder.dechord(midi_obj)
    markers = []
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = num2pitch[chord.root_pc] + '_' + chord.quality + '_' + num2pitch[chord.bass_pc]
        else:
            chord_text = 'N_N_N'
        markers.append(Marker(time=int(cidx * 480), text=chord_text))

    # dedup
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)

    # --- global properties --- #
    # global tempo
    tempos = [b.tempo for b in midi_obj.tempo_changes][:40]
    tempo_median = np.median(tempos)
    global_bpm = int(tempo_median)
    # print(' > [global] bpm:', global_bpm)

    # === save === #
    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    # markers
    midi_obj_out.markers = dedup_chords
    midi_obj_out.markers.insert(0, Marker(text='global_bpm_' + str(int(global_bpm)), time=0))

    # save
    midi_obj_out.instruments[0].name = 'piano'
    midi_obj_out.dump(path_outfile)


# midi2corpus

# ================================================== #
#  Configuration                                     #
# ================================================== #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0}
MIN_BPM = 40
MIN_VELOCITY = 40
NOTE_SORTING = 1  # 0: ascending / 1: descending

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 64 + 1, dtype=int)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60 + 1, dtype=int)
DEFAULT_DURATION_BINS = np.arange(
    BEAT_RESOL / 8, BEAT_RESOL * 8 + 1, BEAT_RESOL / 8)


def midi2corpus_one(path_midi, path_outfile):
    # if not os.path.exists(path_midi):
    #     return
    # if os.path.exists(path_outfile):
    #     return

    print("corpus_one:", path_midi)
    # --- load --- #
    midi_obj = miditoolkit.midi.parser.MidiFile(path_midi)

    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # skip
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx = instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] != 'global' and \
                'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    gobal_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
                marker.text.split('_')[1] == 'bpm':
            gobal_bpm = int(marker.text.split('_')[2])

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(np.round(first_note_time / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL  # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    # print(' > offset:', offset)
    # print(' > last_bar:', last_bar)

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[
                np.argmin(abs(DEFAULT_VELOCITY_BINS - note.velocity))]
            note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS - note.shift))]

            # duration
            note_duration = note.end - note.start
            if note_duration > BAR_RESOL:
                note_duration = BAR_RESOL
            ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
            note.duration = ntick_duration

            # append
            note_grid[quant_time].append(note)

        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time = 0 if chord.time < 0 else chord.time
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time
        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        # quantize
        label.time = label.time - offset * BAR_RESOL
        label.time = 0 if label.time < 0 else label.time
        quant_time = int(np.round(label.time / TICK_RESOL) * TICK_RESOL)

        # append
        label_grid[quant_time] = [label]

    # process global bpm
    gobal_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - gobal_bpm))]

    # collect
    song_data = {
        'notes': intsr_gird,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': gobal_bpm,
            'last_bar': last_bar,
        }
    }

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    print(song_data.keys())
    pickle.dump(song_data, open(path_outfile, 'wb'))



def process_one(mp3_path, midi_path, output_path, file_name, input_midi_suffix='midi'):
    path_midi_input = os.path.join(midi_path, file_name + '.' + input_midi_suffix)
    if mp3_path is not None:
        path_audio_input = os.path.join(mp3_path, file_name + '.mp3')
        path_midi_synchronized_output = os.path.join(os.path.join(output_path, "midi_synchronized"), file_name + '.midi')
    else:
        path_audio_input = None
        path_midi_synchronized_output = path_midi_input
    path_midi_analyzed_output = os.path.join(os.path.join(output_path, "midi_analyzed"), file_name + '.midi')
    path_corpus_output = os.path.join(os.path.join(output_path, "corpus"), file_name + '.pkl')

    try:
        if path_audio_input is not None:
            synchronize_one(path_midi_input, path_audio_input, path_midi_synchronized_output)
        analyze_one(path_midi_synchronized_output, path_midi_analyzed_output)
        midi2corpus_one(path_midi_analyzed_output, path_corpus_output)
    except Exception as ex:
        print(ex)
        print("file {} process failed!".format(file_name))


def main():
    parser = argparse.ArgumentParser(description='Process the transcripted midi file to corpus.')
    parser.add_argument('--mp3_path', default=None, help='mp3 path')
    parser.add_argument('--midi_path', help='midi path')
    parser.add_argument('--input_midi_suffix', default="midi", help='input midi suffix')
    parser.add_argument('--output_path', help='output path')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_path, "midi_synchronized"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "midi_analyzed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "corpus"), exist_ok=True)

    # midi_file_names = traverse_midi_dir(args.midi_path, args.input_midi_suffix)
    midi_file_names = traverse_midi_dir_recurrence(args.midi_path, args.input_midi_suffix)

    print(len(midi_file_names))

    #collect
    data = []
    for fn in midi_file_names:
        path = os.path.join(args.output_path,'corpus',fn+'.pkl')
        # print(path)
        if(os.path.exists(path)):
            print('skip')
            continue
        else : 
            data.append([args.mp3_path, args.midi_path, args.output_path, fn, args.input_midi_suffix])
    # data = []
    # for fn in midi_file_names:
    #     data.append([args.mp3_path, args.midi_path, args.output_path, fn, args.input_midi_suffix])

    # run, multi-thread
    # exit()
    pool = mp.Pool()
    pool.starmap(process_one, data)


def debug_main():
    process_one(None, "../dataSet/PoP909_Pure/midis/001.mid", "../dataSet/PoP909_Pure/nomomo",'001', "mid")


if __name__ == '__main__':
    debug_main()
    # main()

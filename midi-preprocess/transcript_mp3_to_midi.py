import os
import argparse
import torch
import time
from tqdm import tqdm

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio


def inference(args, transcriptor, audio_path, output_midi_path, print_log=True):
    """Inference template.

    Args:
      model_type: str
      audio_path: str
      cuda: bool
    """

    # Arugments & parameters
    # audio_path = args.audio_path
    # output_midi_path = args.output_midi_path

    # Load audio
    if print_log:
        print(audio_path)

    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

    """device: 'cuda' | 'cpu'
    checkpoint_path: None for default path, or str for downloaded checkpoint path.
    """

    # Transcribe and write out to MIDI file
    transcribe_time = time.time()
    transcribed_dict = transcriptor.transcribe(audio, output_midi_path)
    if print_log:
        print('Transcribe time: {:.3f} s'.format(time.time() - transcribe_time))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_midi_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--dir_mode', action='store_true', default=False)

    args = parser.parse_args()

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    # Transcriptor
    transcriptor = PianoTranscription(device=device, checkpoint_path=args.checkpoint_path)

    if args.dir_mode:
        for f in tqdm(os.listdir(args.audio_path)):
            if not f.endswith(".mp3"):
                continue
            fn = os.path.basename(f)
            output_midi_path = os.path.join(args.output_midi_path, fn+ ".midi")
            if os.path.exists(output_midi_path):
                continue
            inference(args, transcriptor, os.path.join(args.audio_path, f), output_midi_path)
    else:
        audio_path = args.audio_path
        output_midi_path = args.output_midi_path
        inference(args, transcriptor, audio_path, output_midi_path)

if __name__ == '__main__':
    main()

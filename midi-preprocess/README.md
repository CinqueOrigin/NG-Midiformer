After download allthe dataset:

1.midi to corpus
2.to events of remi or cp
3.to word for remi or cp

we utilize the unicode to represent the music tokens
tip: you need to download ckpt to /ckpt directory from https://zenodo.org/records/4034264
# Piano DataSet for pre-training

python preprocess_to_corpus.py --midi_path dataSet/PianoDataset/midis --input_midi_suffix midi --output_path dataSet/PianoDataset

python corpus_to_remi_events.py --corpus_path dataSet/PianoDataset/corpus --output_path dataSet/PianoDataset/eventsREMI
python remi_events_to_words.py --events_path dataSet/PianoDataset/eventsREMI --words_path dataSet/PianoDataset/wordsREMI

python corpus_to_cp_events.py --corpus_path dataSet/PianoDataset/corpus --output_path dataSet/PianoDataset/eventsCP
python cp_events_to_words.py --events_path dataSet/PianoDataset/eventsCP --words_path dataSet/PianoDataset/wordsCP

python Merge_cp_words_to_Unicode.py --words_path dataSet/PianoDataset/wordsCP --output_path dataSet/PianoDataset/unicodesCP --dict_file dataSet/PianoDataset/unicodesCP/dictionary.pkl

# Pianist8
python preprocess_to_corpus.py --midi_path dataSet/joann8512-Pianist8-ab9f541/midi --input_midi_suffix mid --output_path dataSet/joann8512-Pianist8-ab9f541

python corpus_to_remi_events.py --corpus_path dataSet/joann8512-Pianist8-ab9f541/corpus --output_path dataSet/joann8512-Pianist8-ab9f541/eventsREMI
python remi_events_to_words.py --events_path dataSet/joann8512-Pianist8-ab9f541/eventsREMI --words_path dataSet/joann8512-Pianist8-ab9f541/wordsREMI

python corpus_to_cp_events.py --corpus_path dataSet/joann8512-Pianist8-ab9f541/corpus --output_path dataSet/joann8512-Pianist8-ab9f541/eventsCP
python cp_events_to_words.py --events_path dataSet/joann8512-Pianist8-ab9f541/eventsCP --words_path dataSet/joann8512-Pianist8-ab9f541/wordsCP

python cp_words_to_Unicode.py --words_path dataSet/joann8512-Pianist8-ab9f541/wordsCP --output_path dataSet/joann8512-Pianist8-ab9f541/unicodesCP --dict_file dataSet/FinalDataSet/unicodesCP/dictionary.pkl

python Merge_cp_words_to_Unicode.py --words_path dataSet/joann8512-Pianist8-ab9f541/wordsCP --output_path dataSet/joann8512-Pianist8-ab9f541/unicodesCP

# EMOPIA
python preprocess_to_corpus.py --midi_path dataSet/EMOPIA_1.0/midis --input_midi_suffix mid --output_path dataSet/EMOPIA_1.0

python corpus_to_remi_events.py --corpus_path dataSet/EMOPIA_1.0/corpus --output_path dataSet/EMOPIA_1.0/eventsREMI
python remi_events_to_words.py --events_path dataSet/EMOPIA_1.0/eventsREMI --words_path dataSet/EMOPIA_1.0/wordsREMI

python corpus_to_cp_events.py --corpus_path dataSet/EMOPIA_1.0/corpus --output_path dataSet/EMOPIA_1.0/eventsCP
python cp_events_to_words.py --events_path dataSet/EMOPIA_1.0/eventsCP --words_path dataSet/EMOPIA_1.0/wordsCP

python cp_words_to_Unicode.py --words_path dataSet/EMOPIA_1.0/wordsCP --output_path dataSet/EMOPIA_1.0/unicodesCP --dict_file dataSet/FinalDataSet/unicodesCP/dictionary.pkl

python Merge_cp_words_to_Unicode.py --words_path dataSet/EMOPIA_1.0/wordsCP --output_path dataSet/EMOPIA_1.0/unicodesCP


# POP909 DataSet

python preprocess_to_corpus.py --midi_path dataSet/PoP909_Pure/midis --input_midi_suffix mid --output_path dataSet/PoP909_Pure

python corpus_to_remi_events.py --corpus_path dataSet/PianoDataset/corpus --output_path dataSet/PianoDataset/eventsREMI
python remi_events_to_words.py --events_path dataSet/PianoDataset/eventsREMI --words_path dataSet/PianoDataset/wordsREMI

python corpus_to_cp_events.py --corpus_path dataSet/PoP909_Pure/corpus --output_path  dataSet/PoP909_Pure/eventsCP
python cp_events_to_words.py --events_path dataSet/PoP909_Pure/eventsCP --words_path dataSet/PoP909_Pure/wordsCP

python cp_words_to_Unicode.py --words_path dataSet/PoP909_Pure/wordsCP --output_path dataSet/PoP909_Pure/UnicodesCP --dict_file dataSet/PianoDataset/unicodesCP/dictionary.pkl

python Merge_cp_words_to_Unicode.py --words_path dataSet/PoP909_Pure/wordsCP  --output_path dataSet/PoP909_Pure/UnicodesCP --dict_file dataSet/PianoDataset/unicodesCP/dictionary.pkl

# GTZAN
python cp_words_to_Unicode.py --words_path dataSet/genre/wordsCP --output_path dataSet/genre/UnicodesCP --dict_file dataSet/PianoDataset/unicodesCP/dictionary.pkl

# Nottingham
python cp_words_to_Unicode.py --words_path dataSet/Nottingham/Dataset/wordsCP --output_path dataSet/Nottingham/Dataset/UnicodesCP --dict_file dataSet/PianoDataset/unicodesCP/dictionary.pkl



# CP4 pretrain
python npy2cp4Unicode.py --input_path dataSet/PianoDataset/cp4 --output_path dataSet/PianoDataset/cp4/mergedpretrain.txt --CP_dict ~/MIDI-BERT/data_creation/prepare_data/dict/CP.pkl --mode pretrain
# CP4 downstream token
python npy2cp4Unicode.py --input_path dataSet/PoP909_Pure/cp4/velocity/pop909_train.npy --input_ans dataSet/PoP909_Pure/cp4/velocity/pop909_train_velans.npy --output_path dataSet/PoP909_Pure/cp4/velocity/train.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode token

python npy2cp4Unicode.py --input_path dataSet/PoP909_Pure/cp4/velocity/pop909_valid.npy --input_ans dataSet/PoP909_Pure/cp4/velocity/pop909_valid_velans.npy --output_path dataSet/PoP909_Pure/cp4/velocity/valid.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode token

python npy2cp4Unicode.py --input_path dataSet/PoP909_Pure/cp4/melody/pop909_train.npy --input_ans dataSet/PoP909_Pure/cp4/melody/pop909_train_melans.npy --output_path dataSet/PoP909_Pure/cp4/melody/train.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode token

python npy2cp4Unicode.py --input_path dataSet/PoP909_Pure/cp4/melody/pop909_valid.npy --input_ans dataSet/PoP909_Pure/cp4/melody/pop909_valid_melans.npy --output_path dataSet/PoP909_Pure/cp4/melody/valid.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode token
# CP4 downstream seq
python npy2cp4Unicode.py --input_path dataSet/joann8512-Pianist8-ab9f541/cp4/composer_train.npy --input_ans dataSet/joann8512-Pianist8-ab9f541/cp4/composer_train_ans.npy --output_path dataSet/joann8512-Pianist8-ab9f541/cp4/train.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/joann8512-Pianist8-ab9f541/cp4/composer_valid.npy --input_ans dataSet/joann8512-Pianist8-ab9f541/cp4/composer_valid_ans.npy --output_path dataSet/joann8512-Pianist8-ab9f541/cp4/valid.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/EMOPIA_1.0/cp4/emopia_train.npy --input_ans dataSet/EMOPIA_1.0/cp4/emopia_train_ans.npy --output_path dataSet/EMOPIA_1.0/cp4/train.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/EMOPIA_1.0/cp4/emopia_valid.npy --input_ans dataSet/EMOPIA_1.0/cp4/emopia_valid_ans.npy --output_path dataSet/EMOPIA_1.0/cp4/valid.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/genre/cp4/GTZAN_train.npy --input_ans dataSet/genre/cp4/GTZAN_train_ans.npy --output_path dataSet/genre/cp4/train.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/genre/cp4/GTZAN_valid.npy --input_ans dataSet/genre/cp4/GTZAN_valid_ans.npy --output_path dataSet/genre/cp4/valid.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/Nottingham/Dataset/cp4/Nottingham_train.npy --input_ans dataSet/Nottingham/Dataset/cp4/Nottingham_train_ans.npy --output_path dataSet/Nottingham/Dataset/cp4/train.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq

python npy2cp4Unicode.py --input_path dataSet/Nottingham/Dataset/cp4/Nottingham_valid.npy --input_ans dataSet/Nottingham/Dataset/cp4/Nottingham_valid_ans.npy --output_path dataSet/Nottingham/Dataset/cp4/valid.json --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode seq



# UC4
python npy2cp4Unicode.py --input_path dataSet/joann8512-Pianist8-ab9f541/cp4/train.json --input_ans dataSet/joann8512-Pianist8-ab9f541/cp4/valid.json --output_path dataSet/joann8512-Pianist8-ab9f541/uc4/ --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode convert

python npy2cp4Unicode.py --input_path dataSet/EMOPIA_1.0/cp4/train.json --input_ans dataSet/EMOPIA_1.0/cp4/valid.json --output_path dataSet/EMOPIA_1.0/uc4 --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode convert

python npy2cp4Unicode.py --input_path dataSet/genre/cp4/train.json --input_ans dataSet/genre/cp4/valid.json --output_path dataSet/genre/uc4/ --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode convert

python npy2cp4Unicode.py --input_path dataSet/Nottingham/Dataset/cp4/train.json --input_ans dataSet/Nottingham/Dataset/cp4/valid.json --output_path dataSet/Nottingham/Dataset/uc4 --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode convert

python npy2cp4Unicode.py --input_path dataSet/PoP909_Pure/cp4/velocity/train.json --input_ans dataSet/PoP909_Pure/cp4/velocity/valid.json --output_path dataSet/PoP909_Pure/uc4/velocity --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode convert

python npy2cp4Unicode.py --input_path dataSet/PoP909_Pure/cp4/melody/train.json --input_ans dataSet/PoP909_Pure/cp4/melody/valid.json --output_path dataSet/PoP909_Pure/uc4/melody --dict_file dataSet/PianoDataset/cp4dict2unicode.pkl --mode convert

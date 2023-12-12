

# N-Gram Unsupervised Compoundation and Feature Injection for Better Symbolic Music Understanding

Our paper has been accepted by AAAI2024 conference! Congratulations!

## Abstract
In this paper, we propose a novel method, NG-Midiformer, for understanding symbolic music sequences that leverages the N-gram approach. Our method involves first processing music pieces into word-like sequences with our proposed unsupervised compoundation, followed by using our N-gram Transformer encoder, which can effectively incorporate N-gram information to enhance the primary encoder part for better understanding of music sequences. The pre-training process on large-scale music datasets enables the model to thoroughly learn the N-gram information contained within music sequences, and subsequently apply this information for making inferences during the fine-tuning stage. Experiment on various datasets demonstrate the effectiveness of our method and achieved state-of-the-art performance on a series of music understanding downstream tasks.

## Getting Started

### 1.Download dataset
| Dataset                       | USAGE\&URL        |
|-------------------------------|--------------|
| Pop1K7 (Hsiao et al. 2021)    | [Pre-training](https://github.com/YatingMusic/compound-word-transformer) |
| ADL Piano (Ferreira, Lelis, and Whitehead 2020b) | [Pre-training](https://github.com/lucasnfe/adl-piano-midi) |
| GIantMIDI-Piano (Kong et al. 2020) | [Pre-training](https://github.com/bytedance/GiantMIDI-Piano) |
| Maestro (Hawthorne et al. 2019) | [Pre-training](https://magenta.tensorflow.org/datasets/maestro) |
| POP909 (Wang* et al. 2020)     | [Melody Extraction&Velocity Prediction](https://github.com/music-x-lab/POP909-Dataset) |
| Nottingham                     | [Dance Classification](https://github.com/jukedeck/nottingham-dataset) |
| Pianist8 (joann8512 2021)      | [Composer Classification](https://zenodo.org/records/5089279) |
| EMOPIA (Hung et al. 2021)      | [Emotion Classification](https://annahung31.github.io/EMOPIA/) |
| GTZAN (Sturm 2013)             | [Genre Classification](https://github.com/chittalpatel/Music-Genre-Classification-GTZAN) |


1.Download all the datasets above to the directory:dataSet
2.each directory in the directory dataSet corresponds to a dataset (downstream),and each directory contains a directory  which contains all the midi files in the directory(e.g. "midis" directory)  
e.g.:
```
wouuyoauin@server:~/dataSet/$ find . -maxdepth 1 -type d
.
./GiantMIDI-piano
./joann8512-Pianist8-ab9f541
./genre
./__MACOSX
./EMOPIA_1.0
./PianoDataset
./POP909
./ADL
./Nottingham
./MAESTRO
```
3.the "PianoDataset" directory's "midis" directory contains all 4 datasets' midi files for pre-training
e.g.:
```
wouuyoauin@server-moon:~/dataSet/PianoDataset/midis$ find . -maxdepth 1 -type d
.
./GiantMIDI-piano
./maestro-v3.0.0
./adl-piano-midi
./Pop1K7
```
### 2. create env
```
pip install -r requirments.txt
```

### 3.Convert midi dataset to CP sequences
cd midi-preprocess
see midi-preprocess/README.md
### 4.Convert CP sequences to UCW sequences
cd UnsupervisedCompoundation
see UnsupervisedCompoundation/README.md
### 5.run pre-training or download pre-traininged checkpoints,and fine-tuning on downstream datasets
cd main
see main/README.md




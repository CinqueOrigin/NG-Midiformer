code references from : https://github.com/glample/fastBPE

cd ./fastBPE to install fastBPE


# learn vocab files from pre-training dataset for CP-7 and convert to ucw-7
learn vocab files from pre-training dataset
```python
python learnBPE.py --txt_file dataSet/PianoDataset/unicodesCP/mergeUnicodeFilePiano.txt --outputCodes_file dataSet/PianoDataset/unicodesCP/codes_1000_Piano --vocab_size 1000
```
convert CP-7 sequences to  ucw-7 sequences for pre-training dataset
```python
python apply_fastBPE.py --txt_file dataSet/PianoDataset/unicodesCP/mergeUnicodeFilePiano.txt  --output_dir dataSet/PianoDataset/subUnicodesCP --codes_file dataSet/PianoDataset/unicodesCP/codes_1000_Piano
```

convert CP-7 sequences to ucw-7 sequences for Pianist8
```python
python apply_fastBPE.py --txt_file dataSet/joann8512-Pianist8-ab9f541/unicodesCP --output_dir dataSet/joann8512-Pianist8-ab9f541/subUnicodesCP --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-7 sequences to ucw-7 sequences for EMOPIA
```python
python apply_fastBPE.py --txt_file dataSet/EMOPIA_1.0/unicodesCP --output_dir dataSet/EMOPIA_1.0/subUnicodesCP --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-7 sequences to ucw-7 sequences for GTZAN
```python
python apply_fastBPE.py --txt_file dataSet/genre/unicodesCP --output_dir dataSet/genre/subUnicodesCP --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-7 sequences to ucw-7 sequences for Nottingham
```python
python apply_fastBPE.py --txt_file dataSet/Nottingham/Dataset/unicodesCP --output_dir  dataSet/Nottingham/Dataset/subUnicodesCP --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-7 sequences to ucw-7 sequences for Pop909_Velocity
```python
python apply_fastBPE.py --txt_file dataSet/PoP909_Pure/uc4/velocity/txt --output_dir dataSet/PoP909/uc4/velocity/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```

# learn vocab files from pre-training dataset for CP-4 and convert to ucw-4
learn vocab files from pre-training dataset
```python
python learnBPE.py --txt_file dataSet/PianoDataset/cp4/mergedpretrain.txt --outputCodes_file dataSet/PianoDataset/cp4/codes_1000_Piano --vocab_size 800
```
convert CP-4 sequences to ucw-4 sequences for pre-training dataset
```python
python apply_fastBPE.py --txt_file dataSet/PianoDataset/cp4/mergedpretrain.txt --output_dir dataSet/PianoDataset/cp4/mergeUnicodeFilePianoBped.txt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-4 sequences to ucw-4 sequences for Pianist8
```python
python apply_fastBPE.py --txt_file dataSet/joann8512-Pianist8-ab9f541/uc4/txt --output_dir dataSet/joann8512-Pianist8-ab9f541/uc4/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-4 sequences to ucw-4 sequences for EMOPIA
```python
python apply_fastBPE.py --txt_file dataSet/EMOPIA_1.0/uc4/txt --output_dir dataSet/EMOPIA_1.0/uc4/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-4 sequences to ucw-4 sequences for GTZAN
```python
python apply_fastBPE.py --txt_file dataSet/genre/uc4/txt --output_dir dataSet/genre/uc4/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-4 sequences to ucw-4 sequences for Nottingham
```python
python apply_fastBPE.py --txt_file dataSet/Nottingham/Dataset/uc4/txt --output_dir  dataSet/Nottingham/Dataset/uc4/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-4 sequences to ucw-4 sequences for Pop909_Velocity
```python
python apply_fastBPE.py --txt_file dataSet/PoP909_Pure/uc4/velocity/txt --output_dir dataSet/PoP909/uc4/velocity/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```
convert CP-4 sequences to ucw-4 sequences for Pop909_Melody
```python
python apply_fastBPE.py --txt_file dataSet/PoP909_Pure/uc4/melody/txt --output_dir dataSet/PoP909/uc4/melody/subtxt --codes_file dataSet/PianoDataset/cp4/codes_1000_Piano
```


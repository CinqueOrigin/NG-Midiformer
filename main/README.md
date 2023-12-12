
# Pre-Training with NG-Midiformer on UCW-7
1) run pre-training with the following steps:
1.please download the model from:IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese': 'https://huggingface.co/IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese to diretory ZEN2-345M
2.After get the pre-training dataset,please get vocab.txt and ngram.txt files by running utils/GetVocabs.py and utils/CreateN-gram.py respectively, and then replace the corrsponding file in ZEN2-345M directory
3.Run pre-training with the following command with 4gpus(you can change --nproc_per_node to train with other number of GPUS)
```
python -m torch.distributed.launch --nproc_per_node 4 PreTrain.py --model_name_or_path ZEN2-345M  --cache_dir ./cache --train_file ../dataSet/PianoDataset/unicodesCP/mergeUnicodeFilePiano.txt  --validation_split_percentage 5 --line_by_line false --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --do_train true --do_eval true --output_dir ./outputUCW7 --logging_dir ./outputUCW7 --max_seq_length 512 --num_train_epochs 10 --do_eval true --save_steps 5000
```
2) or you can download our pre-trained checkpoint with the following checkpoints:




If you do this, please build the dataset according to our settings and initialization to prevent situations where the symbolic music vocabulary does not correspond.
# evaluation on downstream tasks on UCW-7
fine-tune on EMOPIA dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW7  --output_save_path ./outputEmoUCW7 --data_dir ../dataSet/EMOPIA_1.0 --train_data train_emo_lang.json --valid_data val_emo_lang.json --train_batchsize 6 --valid_batchsize 16 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max  --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 15 --gpus 4 --check_val_every_n_epoch 1 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 4 --dirpath ./outputEmoUCW7
```
fine-tune on Pianist8 dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW7  --output_save_path ./outputComUCW7 --data_dir ../dataSet/joann8512-Pianist8-ab9f541 --train_data train_composer_long.json --valid_data val_composer_long.json --train_batchsize 6 --valid_batchsize 16 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max  --check_val_every_n_epoch 1 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 15 --gpus 4  --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 8 --dirpath ./outputComUCW7
```
fine-tune on GTZAN dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW7  --output_save_path ./outputGenreMelUCW7 --data_dir ../dataSet/genre --train_data train_genre_long_.json --valid_data val_genre_long_NOPBPE.json --train_batchsize 6 --valid_batchsize 16 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 10 --gpus 4 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 10 --dirpath ./outputGenreMelUCW7 
```
fine-tune on Nottingham dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW7  --output_save_path ./outputDanceUCW7 --data_dir ../dataSet/Nottingham/Dataset --train_data train_dance_long_.json --valid_data val_dance_long_.json --train_batchsize 6 --valid_batchsize 16 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 20 --gpus 4 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 14 --dirpath ./outputDanceUCW7
```


# Pre-Training with NG-Midiformer on UCW-4
1) run pre-training with the following steps:
1.please download the model from:IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese': 'https://huggingface.co/IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese to diretory ZEN2-345M
2.After get the pre-training dataset,please get vocab.txt and ngram.txt files by running utils/GetVocabs.py and utils/CreateN-gram.py respectively, and then replace the corrsponding file in ZEN2-345M directory
3.Run pre-training with the following command with 4gpus(you can change --nproc_per_node to train with other number of GPUS)
```
python -m torch.distributed.launch --nproc_per_node 4 PreTrain.py --model_name_or_path ZEN2-345M  --cache_dir ./cache --train_file ../dataSet/PianoDataset/cp4/uc4/mergeUnicodeFilePiad.txt  --validation_split_percentage 5 --line_by_line false --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --do_train true --do_eval true --output_dir ./outputUCW4 --logging_dir ./outputUCW4 --max_seq_length 512 --num_train_epochs 10 --do_eval true --save_steps 5000
```
1) or you can download our pre-trained checkpoint with the following checkpoints:

token classification:
fine-tune on POP909 dataset on Velocity
```
python NGMidiformerForTokenClassification.py --pretrained_model_path outputUCW4  --output_save_path ./outputVelUCW4 --data_dir ../dataSet/PoP909/uc4/velocity --train_data train.json --valid_data valid.json --train_batchsize 8 --valid_batchsize 8 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.1 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max  --check_val_every_n_epoch 1 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 15 --gpus 4 --save_weights_only False --task_name Velocity --dirpath ./outputVelUCW4 
```
fine-tune on POP909 dataset on Melody
```
python NGMidiformerForTokenClassification.py --pretrained_model_path outputUCW4  --output_save_path ./outputMelUCW4 --data_dir ../dataSet/PoP909/uc4/melody --train_data train.json --valid_data valid.json --train_batchsize 8 --valid_batchsize 8 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.1 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max  --check_val_every_n_epoch 1 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 15 --gpus 4 --save_weights_only False --task_name Melody --dirpath ./outputMelUCW4 
```
sequence classification: 
fine-tune on Pianist8 dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW4 --output_save_path ./outputComposerUCW4 --data_dir ../dataSet/joann8512-Pianist8-ab9f541/uc4 --train_data train.json --valid_data valid.json --train_batchsize 6 --valid_batchsize 8 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max --every_n_train_steps 400 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 15 --gpus 4 --check_val_every_n_epoch 1 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 8 --dirpath ./outputComposerUCW4
```
fine-tune on EMOPIA dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW4  --output_save_path ./outputEmotionUCW4 --data_dir ../dataSet/EMOPIA_1.0/uc4 --train_data train.json --valid_data valid.json --train_batchsize 6 --valid_batchsize 8 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max --every_n_train_steps 400 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 25 --gpus 4 --check_val_every_n_epoch 1 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 4 --dirpath ./outputEmotionUCW4 
```
fine-tune on GTZAN dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW4 --data_dir ../dataSet/genre/uc4 --train_data train.json --valid_data valid.json --train_batchsize 6 --valid_batchsize 8 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max --every_n_train_steps 400 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 20 --gpus 4 --check_val_every_n_epoch 1 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 10 --dirpath ./outputGenreUCW4  --output_save_path ./outputGenreUCW4
```
fine-tune on Nottingham dataset
```
python NGMidiformerForSequenceClassification.py --pretrained_model_path outputUCW4   --data_dir ../dataSet/Nottingham/Dataset/uc4 --train_data train.json --valid_data valid.json --train_batchsize 6 --valid_batchsize 8 --max_seq_length 512 --texta_name text --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.01 --monitor val_acc --save_top_k 3 --mode max --every_n_train_steps 400 --filename model-{epoch:02d}-{val_acc:.4f} --max_epochs 20 --gpus 4 --check_val_every_n_epoch 1 --save_on_train_epoch_end False --save_weights_only False --task_name Composer --num_labels 14 --dirpath ./outputDanceUCW4  --output_save_path ./outputDanceUCW4 
```
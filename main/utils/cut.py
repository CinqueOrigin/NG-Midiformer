import re

txt = '--model_name_or_path .  --cache_dir ./cache --train_file ~/workSpace/musicbert/SuboutputOctuple_30000/mergedTrainOctuple.txt --validation_file ~/workSpace/musicbert/SuboutputOctuple/mergedValidOctuple.txt --validation_split_percentage 5 --line_by_line false --per_device_train_batch_size 2  --do_train true --do_eval true --output_dir ./outputOctuple_30000 --max_seq_length 1024 --num_train_epochs 1 --do_eval true --save_steps 2000 --fp16 --fp16_full_eval true'
list1 = re.split(r"[ ]+", txt)
count = 0
for itm in list1:
    print('"'+itm+'"',end=',')
    count+=1
    if count%4 == 0:
        print('')
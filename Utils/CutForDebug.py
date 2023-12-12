import re
def cutForDebug(txt=None):
    if txt is None:
        txt = '--model_name_or_path ~/data/Work/ZEN2-345M-CP4  --cache_dir ./cache --train_file ~/data/Work/dataset/mergedpretrain.txt  --validation_split_percentage 5 --line_by_line false --per_device_train_batch_size 8 --per_device_eval_batch_size 32 --do_train true --do_eval true --output_dir ./outputCP4NoBPE --logging_dir ./outputCP4NoBPE --max_seq_length 512 --num_train_epochs 10 --do_eval true --save_steps 5000'
    list1 = re.split(r"[ ]+", txt)
    count = 0
    for itm in list1:
        print('"'+itm+'"',end=',')
        count+=1
        if count%4 == 0:
            print('')

if __name__ == '__main__':
    cutForDebug()
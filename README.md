# Applied ML project

Create dataset automatically

```
python dataset.py
```

```
usage: Create  [-h] [--img_dir IMG_DIR] [--save_dir SAVE_DIR]
               [--num_class NUM_CLASS]

What the program does

options:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR
  --save_dir SAVE_DIR
  --num_class NUM_CLASS
                        Number of most appeared classes use to train
```


Run train

```
python train.py
```

```
usage: train.py [-h] [--data_dir DATA_DIR] [--pretrained PRETRAINED]
                [--freeze_backbone FREEZE_BACKBONE] [--optimizer OPTIMIZER]
                [--batch_size BATCH_SIZE] [--seed SEED] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE] [--summary SUMMARY]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR
  --pretrained PRETRAINED
  --freeze_backbone FREEZE_BACKBONE
  --optimizer OPTIMIZER
  --batch_size BATCH_SIZE
  --seed SEED
  --epochs EPOCHS
  --learning_rate LEARNING_RATE
  --summary SUMMARY
```
Read argparse for more details.

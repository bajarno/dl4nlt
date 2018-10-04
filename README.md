# Abstractive Headline Generation for News Articles

This repository contains the code for the project done as part of the course 'Deep learning for language technology'. All code for training and evaluation is located in the `code` folder.

## Pre-processing

TODO

## Training

To train the NAMAS model with standard hyperparameters:
```
python train.py
```

To train the sequence-to-sequence model with standard hyperparameters:
```
python train_s2s.py
```

For custom hyperparameters, add `--help` to your command to view all options.

## Testing
After training, models are saved in the `model_checkpoints` directory. To evaluate the NAMAS model:

``` 
python test.py --model_file <checkpoint_name>
```
Or for the sequence-to-sequence model:
``` 
python test_s2s.py --model_file <checkpoint_name>
```

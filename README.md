# Abstractive Headline Generation for News Articles

This repository contains the code for the project done as part of the course 'Deep learning for language technology'. All code for training and evaluation is located in the `code` folder. All code related to pre-processing is located in the `data` folder.

## Pre-processing

First the dataset files need to be retrieved from [Kaggle](https://www.kaggle.com/snapcrack/all-the-news) and placed in `data/kaggle`. To parse the dataset files:
```
python kaggleparser.py
```

To generate a subword dataset, a subword model needs to be generated before the data can be tokenized:
```
python subword_model_trainer.py
```

To preprocess the parsed file:
```
python datapreprocessory.py
```


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
Be sure to use the same hyperparameter options the model was trained with.

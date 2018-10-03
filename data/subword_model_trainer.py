import sentencepiece

input_file = 'kaggle_parsed_1_40_100.csv'
vocab_size = 16000
sentencepiece.SentencePieceTrainer.Train('--input=' + input_file + ' --model_prefix=subword' + str(vocab_size) + ' --vocab_size=' + str(vocab_size))

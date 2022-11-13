file_path = "\Data\Dataset.csv"
save_model_path = "\Data\saved_model\\abusing_detection_1.h5"
mecab_vocab_path = "\Data\\after_mecab.txt"
pretrained_tokenizer_path = "\Data\\tokenizer_model\\vocab.txt"


MODEL_NAME = 'roberta-base'
MAX_LEN = 256

BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = BATCH_SIZE_PER_REPLICA# * strategy.num_replicas_in_sync
EPOCHS = 40
from transformers import (
    PreTrainedTokenizerFast,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments
)
print('load tokenizer')
tokenizer = PreTrainedTokenizerFast.from_pretrained('../bert-base-dv')

train_file = '../../data/dv-corpus-clean-unique.txt'
print('init dataset')
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

config = BertConfig(
    vocab_size=30_000,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)
print('init model')
model = BertForMaskedLM(config)

training_args = TrainingArguments(
    output_dir="bert-base-dv",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)
print('begin training')
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model('bert-base-dv')
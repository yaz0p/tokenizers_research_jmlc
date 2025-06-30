from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tokenizer_tuner import TokenizerTuner
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTConfig


class TurkishTokenizerTuner(TokenizerTuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def clean_sentence(sentence):
        return re.sub(
            r'[^a-zA-Z0-9ğüşöçıĞÜŞÖÇİ\s\n\r`",.;:!?\'"()/@&#%+=*/-]',
            '',
            sentence
        )

    def _batch_iterator(self):
        for i in (self.dataset):
            yield self.clean_sentence(i["text"])


model_name = "HuggingFaceTB/SmolLM2-1.7B"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_tokenizer = AutoTokenizer.from_pretrained(model_name)

if base_tokenizer.pad_token is None:
    base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(base_tokenizer))
raw_dataset = load_dataset(load_dataset("HuggingFaceFW/fineweb-2", 'tur_Latn', split="train", streaming=True))

tuner = TurkishTokenizerTuner(
    model=base_model,
    tokenizer=base_tokenizer,
    dataset=raw_dataset,
    vocab_size=20_000
)

new_tokenizer, tuned_model = tuner.tune()

print(f"\nNew tokenizer: {len(base_tokenizer)}")
print(f"Tokenizer size: {len(new_tokenizer)}")
print(f"Embeddings size: {tuned_model.get_input_embeddings().weight.shape[0]}")

new_tokenizer.save_pretrained("./my_tuned_tokenizer")
tuned_model.save_pretrained("./my_tuned_model")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)

model = get_peft_model(tuned_model, lora_config)

sft_config = SFTConfig(
    output_dir="./results", 
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    save_steps=50,
    logging_steps=50,
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    gradient_checkpointing=True,
    optim='adamw_torch',
    report_to="none",
)


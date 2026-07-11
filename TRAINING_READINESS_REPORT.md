## GO / NO GO Decision

NO GO

## Remaining Blockers

- domain_mental_health_adjacent: 66.54% versus maximum threshold 35.00%
- exact_duplicate_rate: 37.77% versus maximum threshold 0.00%

## Recommended Base Model

Selected model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`

The repository already contains a formal base-model decision selecting `HuggingFaceTB/SmolLM2-1.7B-Instruct`, so this readiness pass respects that lock. Among the sub-2B candidates already considered in the project, SmolLM2 offered the strongest instruction-following tradeoff at the chosen size, uses a straightforward chat template that is well suited to English conversational fine-tuning, and remains comfortably small enough for a 4-bit rank-16 LoRA calibration run on Kaggle T4 GPUs.

## LoRA Hyperparameters

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
learning_rate: 0.0002
warmup_ratio: 0.05
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
fp16: true
bf16: false
max_seq_length: 2048
```

## Estimated VRAM

Estimated requirement: approximately 12 GB VRAM per GPU for a 4-bit QLoRA run with rank 16, alpha 32, sequence length 2048, and per-device batch size 2. Recommended accelerator: Kaggle T4 x2.

## Estimated Kaggle Runtime

Estimated one-epoch wall-clock time on Kaggle T4 x2: approximately 3.4 to 5.1 hours, based on 24463 train samples and an average observed tokenizer length of 450.22 tokens in the 50-sample spot check.

## Dataset Statistics

- Total samples assembled: 27181
- Samples in train / val / test: 24463 / 1359 / 1359
- Language labels use the existing pipeline outputs: `mostly_ascii` and `majority_non_ascii`.

Language distribution:
- majority_non_ascii: 9140 samples (33.63%)
- mostly_ascii: 18041 samples (66.37%)

Domain distribution:
- fitness-coaching-adjacent: 9094 samples (33.46%)
- mental-health-adjacent: 18087 samples (66.54%)

Source contribution:
- hizardev/MentalHealth-Counseling: 18041 samples (66.37%)
- ulysses531/fitness-conversation-dataset: 9140 samples (33.63%)
- Truncation rate at max_seq_length: 0.00%

## Train / Validation / Test Split

- Train: `D:\data\data\final\train.jsonl` | SHA256 `8e1369d33675ad4f7010e2489b297edfdbdb46051c765142b34cd4f11befe640` | 24463 samples
- Val: `D:\data\data\final\val.jsonl` | SHA256 `7fe8cae6e5fdcd8bf46afd0a3fefdc1cb484e5447cf6b64b687152ed0929bf55` | 1359 samples
- Test: `D:\data\data\final\test.jsonl` | SHA256 `fcc6e73b149bc922d368f082763e0538494d92926c670131e00084b650c8765b` | 1359 samples

## Recommended Next Command

```bash
python -m pip install -q peft bitsandbytes trl
python - <<'PY'
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

model_id = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
train_path = 'D:\\data\\data\\final\\train.jsonl'
val_path = 'D:\\data\\data\\final\\val.jsonl'
output_dir = 'D:\\data\\training\\checkpoints\\calibration_lora_run'

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype='float16',
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map='auto',
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    task_type='CAUSAL_LM',
)
train_dataset = load_dataset('json', data_files=train_path, split='train')
eval_dataset = load_dataset('json', data_files=val_path, split='train')
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    num_train_epochs=1,
    logging_steps=25,
    eval_strategy='steps',
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to='none',
    seed=42,
    dataloader_num_workers=2,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    dataset_text_field='text',
    max_seq_length=2048,
)
trainer.train()
trainer.evaluate()
trainer.save_model()
PY
```

## Expected Output Artifacts

- `D:\data\training\checkpoints\calibration_lora_run\adapter_model.safetensors`
- `D:\data\training\checkpoints\calibration_lora_run\adapter_config.json`
- `D:\data\training\checkpoints\calibration_lora_run\trainer_state.json`
- `D:\data\training\checkpoints\calibration_lora_run\training_args.bin`
- `D:\data\training\checkpoints\calibration_lora_run\train_results.json`
- `D:\data\training\checkpoints\calibration_lora_run\eval_results.json`
- `D:\data\training\checkpoints\calibration_lora_run\all_results.json`

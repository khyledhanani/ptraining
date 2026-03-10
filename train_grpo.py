from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset = load_dataset("trl-lib/DeepMath-103K", split="train").select(range(5000))

training_args = GRPOConfig(
    output_dir="grpo-qwen-math-baseline",
    run_name="iid-sampling",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-6,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    max_completion_length=512,
    num_generations=8,
    seed=42,
    report_to="wandb",
    dataloader_num_workers=4,
)

trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=accuracy_reward,
)

trainer.train()
trainer.save_model()

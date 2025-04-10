import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from ctransformers import AutoModelForCausalLM as CTModelForCausalLM

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_stats = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'perplexity': []
        }
    
    def log(self, logs, start_time=None): 
        super().log(logs)
        if 'loss' in logs:
            self.training_stats['train_loss'].append(logs['loss'])
        if 'eval_loss' in logs:
            self.training_stats['eval_loss'].append(logs['eval_loss'])
        if 'learning_rate' in logs:
            self.training_stats['learning_rate'].append(logs['learning_rate'])
        if 'eval_loss' in logs:
            self.training_stats['perplexity'].append(np.exp(logs['eval_loss']))


def plot_training_stats(stats, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制训练损失
    ax1.plot(stats['train_loss'], label='训练损失')
    ax1.set_title('训练损失曲线')
    ax1.set_xlabel('步数')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制验证损失
    ax2.plot(stats['eval_loss'], label='验证损失')
    ax2.set_title('验证损失曲线')
    ax2.set_xlabel('步数')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    # 绘制学习率
    ax3.plot(stats['learning_rate'], label='学习率')
    ax3.set_title('学习率变化曲线')
    ax3.set_xlabel('步数')
    ax3.set_ylabel('学习率')
    ax3.legend()
    ax3.grid(True)
    
    # 绘制困惑度
    ax4.plot(stats['perplexity'], label='困惑度')
    ax4.set_title('困惑度变化曲线')
    ax4.set_xlabel('步数')
    ax4.set_ylabel('困惑度')
    ax4.legend()
    ax4.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f'training_stats_{timestamp}.png'))
    plt.close()

def train_model():
    # 加载模型和tokenizer
    model_path = "/home/wkr/.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-base/snapshots/c919139c3a9b4070729c8b2cca4847ab29ca8d94"

    
    # 加载Hugging Face模型用于训练 - 不使用量化
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用半精度以节省显存
        use_cache=False,
        low_cpu_mem_usage=True,
        use_safetensors=False  # 使用PyTorch格式而不是safetensors
    )
    
    # 启用gradient checkpointing以节省显存
    hf_model.gradient_checkpointing_enable()
    
    # 配置LoRA - 调整参数以适应较小的模型
    lora_config = LoraConfig(
        r=8,  # 减小rank，因为模型较小
        lora_alpha=16,  # 调整alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    hf_model = get_peft_model(hf_model, lora_config)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    train_file = "dataset/final/train.json"
    val_file = "dataset/final/val.json"
    test_file = "dataset/final/test.json"
    
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 打印数据集信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 数据预处理函数
    def preprocess_function(examples):
        prompts = [
            f"Instruction: {instruction}\nInput: \nOutput: {text}"
            for instruction, text in zip(
                examples["instruction"],
                examples["text"]
            )
        ]
        
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=96,  # 减小最大长度以节省显存
            padding="max_length"
        )
        
        return tokenized
    
    # 处理数据集
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 训练参数 - 针对8GB显存优化
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=4,  # 增加batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 减少梯度累积步数
        learning_rate=1e-3,  # 增加学习率
        fp16=True,  # 启用混合精度训练
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        eval_steps=500,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        save_total_limit=2,
        report_to="none"
    )
    
    # 创建训练器
    trainer = CustomTrainer(
        model=hf_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model("./final_model")
    
    # 绘制训练统计图表
    plot_training_stats(trainer.training_stats, "./results")

def check_data_quality(dataset):
    # 检查空值
    null_counts = dataset.isnull().sum()
    print("空值统计:", null_counts)
    
    # 检查文本长度分布
    lengths = [len(text) for text in dataset["instruction"]]
    print(f"平均长度: {sum(lengths)/len(lengths)}")
    print(f"最大长度: {max(lengths)}")
    print(f"最小长度: {min(lengths)}")

if __name__ == "__main__":
    train_model() 
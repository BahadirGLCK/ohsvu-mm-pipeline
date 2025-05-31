import pathlib

# --- Project Root (determined dynamically if possible, or set as needed) ---
# For scripts in /scripts, this would be pathlib.Path(__file__).parent.parent
# For modules in /src, this would be pathlib.Path(__file__).parent.parent
# Assuming scripts will be run from the project root, or paths adjusted accordingly.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve() # Adjust if scripts are run from other dirs

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data" / "inputs" # Base directory for input files
VIDEO_DIR = DATA_DIR / "vlm_videos"
OHS_PROMPT_FILENAME = "OHS_PROMPT_v2.txt"
OHS_PROMPT_PATH = DATA_DIR / OHS_PROMPT_FILENAME
CSV_FILENAME = "gemini_pro_v3_outputs.csv"
CSV_PATH = DATA_DIR / CSV_FILENAME

# --- Experiment Paths ---
EXPERIMENT_ROOT_DIR = PROJECT_ROOT / "experiments"
DATA_SPLIT_DIR_NAME = "data_split"
FINETUNED_MODEL_DIR_NAME = "finetuned_model"
LORA_CHECKPOINT_NAME = "final" # Sub-directory for the final LoRA checkpoint

# --- Model Configuration ---
BASE_MODEL_ID = "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LEN_FINETUNE = 10000
MAX_SEQ_LEN_EVAL = 2000 # Can be same or different from finetuning

# --- Finetuning Hyperparameters ---
RANDOM_SEED = 42
NUM_FRAMES_PER_VIDEO = 25
SAMPLING_STRATEGY = "uniform"  # "uniform" | "stride" | "random" | "index"

# LoRA Configuration (can be a dictionary)
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "finetune_vision_layers": True,
    "finetune_language_layers": True,
    "use_rslora": True,
    "random_state": RANDOM_SEED, # Reuse general random seed or specify another
}

# Training Arguments Configuration (can be a dictionary)
# output_dir will be constructed in the finetuning script using EXPERIMENT_ROOT_DIR
TRAINING_ARGS_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "save_strategy": "steps",
    "save_steps": 10,
    "report_to": "tensorboard",
    "fp16": False,
    "bf16": True,
    "optim": "adamw_8bit",
}

# --- Evaluation Hyperparameters ---
DEVICE = "cuda"
# NUM_FRAMES_PER_VIDEO and SAMPLING_STRATEGY can be reused from finetuning 
# or defined separately if evaluation needs different settings.
# RANDOM_SEED_SAMPLING for evaluation can be same as RANDOM_SEED or different.
EVAL_RANDOM_SEED_SAMPLING = RANDOM_SEED 

# Generation parameters for evaluation
EVAL_GENERATION_CONFIG = {
    "max_new_tokens": 2000,
    "temperature": 0.2,
    "top_p": 0.95,
}

# --- Data Splitting (for finetuning script) ---
TRAIN_SPLIT_RATIO = 0.75
EVAL_SPLIT_RATIO = 0.15
# TEST_SPLIT_RATIO is implicitly 1 - TRAIN_SPLIT_RATIO - EVAL_SPLIT_RATIO

# --- Metrics Configuration (for evaluation script) ---
OBJECT_KEYS_FOR_METRICS = ["forklifts", "pedestrians", "drivers", "trucks"] 
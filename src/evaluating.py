import sys
import json
import pathlib
import re
# import random # random is not directly used in this class, seeding is for global config
import logging
from io import BytesIO

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score
from transformers import AutoProcessor
from unsloth import FastVisionModel
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from .utils import row_to_example
from .experiment_manager import ExperimentManager

class Evaluator:
    """
    Handles the evaluation of a finetuned model against a test dataset.
    This includes loading the model, preparing test data, generating predictions,
    and calculating various performance metrics.

    Attributes:
        global_config: The global configuration module.
        exp_manager (ExperimentManager): Manager for the specific experiment being evaluated.
                                         Its path is already set to the target experiment.
        logger (logging.Logger): Logger instance, typically from ExperimentManager.
        base_model_id (str): Identifier for the base model.
        max_seq_len_eval (int): Maximum sequence length for evaluation.
        device_cfg (str): Device to run evaluation on (e.g., "cuda").
        eval_gen_cfg (dict): Configuration for model's generate method during evaluation.
        metric_obj_keys (list): List of object keys used for calculating count metrics.
        model: The loaded PEFT model for evaluation.
        processor: The processor associated with the model.
        tokenizer: The tokenizer associated with the model.
        test_examples (list): List of prepared examples for the test set.
    """
    def __init__(self, global_config_module, experiment_manager: ExperimentManager):
        """
        Initializes the Evaluator for a specific experiment.

        Args:
            global_config_module: The main configuration module (e.g., src.config).
            experiment_manager: An ExperimentManager instance already configured
                                with the path to the experiment to be evaluated.
        """
        self.global_config = global_config_module
        self.exp_manager = experiment_manager
        self.logger = getattr(experiment_manager, 'logger', logging.getLogger(f"{__name__}.Evaluator"))
        if not self.exp_manager.logger:
            self.logger.warning("ExperimentManager logger not found. Evaluator using fallback logger.")

        # Load config values, prioritizing experiment-specific config, then global, then defaults.
        self.base_model_id = self.exp_manager.get_config_value("BASE_MODEL_ID", main_config_module=self.global_config)
        self.max_seq_len_eval = self.exp_manager.get_config_value("MAX_SEQ_LEN_EVAL", main_config_module=self.global_config)
        self.device_cfg = self.exp_manager.get_config_value("DEVICE", main_config_module=self.global_config)
        self.eval_gen_cfg = self.exp_manager.get_config_value("EVAL_GENERATION_CONFIG", main_config_module=self.global_config, default_value={})
        self.metric_obj_keys = self.exp_manager.get_config_value("OBJECT_KEYS_FOR_METRICS", main_config_module=self.global_config, default_value=[])

        # Initialize attributes
        self.model, self.processor, self.tokenizer = None, None, None
        self.test_examples = []
        self.logger.info(f"Evaluator initialized for experiment: {self.exp_manager.current_experiment_path}")

    def _load_video_names(self, txt_path: pathlib.Path) -> list[str]:
        """Loads a list of video names from a text file (one name per line)."""
        self.logger.debug(f"Loading video names from {txt_path}")
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            self.logger.error(f"Video names file not found: {txt_path}")
            return []

    def _extract_images(self, sample: dict) -> list[Image.Image]:
        """
        Extracts PIL Image objects from a sample dictionary.
        Handles images embedded as paths, bytes, or already PIL.Image objects.
        """
        self.logger.debug(f"Extracting images for sample related to video: {sample.get('video_path', 'Unknown')}")
        imgs = []
        user_content = sample.get("messages", [{}])[0].get("content", [])
        for blk in user_content:
            if blk.get("type") != "image": continue
            img_obj = blk.get("image")
            try:
                if isinstance(img_obj, Image.Image): 
                    imgs.append(img_obj)
                elif isinstance(img_obj, dict) and img_obj.get("bytes"): 
                    imgs.append(Image.open(BytesIO(img_obj["bytes"])).convert("RGB"))
                elif isinstance(img_obj, dict) and img_obj.get("path"): 
                    imgs.append(Image.open(img_obj["path"]).convert("RGB"))
                elif isinstance(img_obj, (str, pathlib.Path)): # Allow direct path strings/objects
                    imgs.append(Image.open(img_obj).convert("RGB"))
            except Exception as e:
                self.logger.warning(f"Could not load image from source {img_obj}: {e}")
        self.logger.debug(f"Extracted {len(imgs)} images.")
        return imgs

    def _generate_reply_for_sample(self, sample: dict, ohs_prompt: str) -> dict:
        """
        Generates a model reply (parsed JSON) for a single evaluation sample.
        
        Args:
            sample (dict): The input sample dictionary, expected to contain image data and video path.
            ohs_prompt (str): The OHS prompt string.

        Returns:
            dict: Parsed JSON from the model's reply, or an empty dict on failure.
        """
        video_p = sample.get('video_path', 'Unknown')
        self.logger.debug(f"Generating reply for sample: {video_p}")
        images = self._extract_images(sample)
        if not images: 
            self.logger.warning(f"No images found or loadable for sample {video_p}, cannot generate reply.")
            return {}
        
        messages_for_model = [{"role": "user", "content": ([{"type": "image", "image": img} for img in images] + 
                                                       [{"type": "text",  "text": ohs_prompt}])}]
        try:
            text = self.processor.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages_for_model) # Assumes this handles PIL images
            batch = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device_cfg)
            
            with torch.inference_mode():
                gen_ids = self.model.generate(**batch, **self.eval_gen_cfg, use_cache=True)
            
            gen_trim = gen_ids[:, batch.input_ids.shape[1]:]
            reply = self.processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            self.logger.debug(f"Generated reply for {video_p} (first 100 chars): {reply[:100]}...")
            
            match = re.search(r"\{[\s\S]*\}", reply)
            if match:
                parsed_json = json.loads(match.group())
                return parsed_json
            else:
                self.logger.warning(f"No JSON-like structure found in reply for {video_p}. Reply: {reply}")
                return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError parsing reply for {video_p}. Reply: {reply}. Error: {e}")
            return {}
        except Exception as e:
            self.logger.exception(f"Unexpected error during reply generation for {video_p}: {e}")
            return {}


    def load_model_for_evaluation(self) -> None:
        """
        Loads the finetuned LoRA model and base model components for evaluation.
        Sets `self.model`, `self.tokenizer`, `self.processor`.
        Exits if the LoRA checkpoint is not found.
        """
        self.logger.info("Loading model for evaluation...")
        lora_model_path = self.exp_manager.get_lora_checkpoint_path(
            self.exp_manager.get_config_value("FINETUNED_MODEL_DIR_NAME", main_config_module=self.global_config),
            self.exp_manager.get_config_value("LORA_CHECKPOINT_NAME", main_config_module=self.global_config)
        )
        self.logger.info(f"Attempting to load LoRA adapter from: {lora_model_path}")
        
        if not lora_model_path.is_dir() or not (lora_model_path / "adapter_config.json").exists():
            self.logger.critical(f"LoRA checkpoint not found or invalid at {lora_model_path}. Cannot proceed with evaluation.")
            sys.exit(1) # Critical error, stop execution
        
        self.logger.info(f"Loading base model: {self.base_model_id} with max_seq_len: {self.max_seq_len_eval} for evaluation.")
        base_model, tokenizer = FastVisionModel.from_pretrained(
            self.base_model_id, load_in_4bit=True, 
            use_gradient_checkpointing="unsloth", # "unsloth" is for training, can be False or removed for pure inference
            max_seq_length=self.max_seq_len_eval
        )
        self.logger.info("Applying PEFT LoRA model to the base model...")
        self.model = PeftModel.from_pretrained(base_model, str(lora_model_path))
        FastVisionModel.for_inference(self.model) # Prepare for inference
        self.model.eval() # Set to evaluation mode
        self.model = self.model.to(self.device_cfg)
        
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
        self.tokenizer = tokenizer # Tokenizer from base model loading
        self.logger.info(f"Model successfully loaded and moved to device '{self.device_cfg}'. Base: {self.base_model_id}, LoRA: {lora_model_path}")

    def prepare_evaluation_data(self, ohs_prompt_text: str) -> None:
        """
        Prepares the test dataset for evaluation.
        Loads test video names from the experiment's data split directory,
        filters the main CSV for these videos, and creates example dicts.
        Sets `self.test_examples`. Exits if critical files are missing.
        
        Args:
            ohs_prompt_text (str): The OHS prompt string.
        """
        self.logger.info("Preparing evaluation data...")
        data_split_dir_name = self.exp_manager.get_config_value("DATA_SPLIT_DIR_NAME", main_config_module=self.global_config)
        video_names_path = self.exp_manager.get_data_split_dir(data_split_dir_name) / "test_video_names.txt"
        
        # CSV_PATH should come from global_config as it's a primary data source path
        csv_gt_path = pathlib.Path(self.global_config.CSV_PATH)

        if not video_names_path.exists():
            self.logger.critical(f"Test video names file not found at {video_names_path}. Cannot prepare evaluation data."); sys.exit(1)
        video_names_to_test = self._load_video_names(video_names_path)
        if not video_names_to_test:
             self.logger.warning(f"No video names loaded from {video_names_path}. Test set will be empty.")
             self.test_examples = []
             return
        self.logger.info(f"Loaded {len(video_names_to_test)} video names for testing from {video_names_path}")
        
        if not csv_gt_path.exists():
            self.logger.critical(f"Ground truth CSV file not found at {csv_gt_path}. Cannot prepare evaluation data."); sys.exit(1)
        
        try:
            raw_df = pd.read_csv(csv_gt_path, header=None, names=["video_name", "gemini_answer"], encoding="utf-8")
            test_df = raw_df[raw_df["video_name"].isin(video_names_to_test)]
        except Exception as e:
            self.logger.critical(f"Failed to load or process ground truth CSV from {csv_gt_path}: {e}"); sys.exit(1)


        if test_df.empty:
            self.logger.warning(f"No matching videos found in CSV ({csv_gt_path}) for the test set defined in {video_names_path}. Evaluation will have no samples.")
            self.test_examples = []
            return
        
        self.test_examples = [row_to_example(r, ohs_prompt_text, global_config=self.global_config) 
                              for _, r in tqdm(test_df.iterrows(), total=len(test_df), desc="Preparing test examples")]
        self.logger.info(f"Prepared {len(self.test_examples)} test examples for evaluation.")

    def collect_predictions_and_ground_truths(self, ohs_prompt_text: str) -> tuple[list, list, list, list, list]:
        """
        Iterates through `self.test_examples`, generates model predictions,
        and extracts ground truth information.
        
        Args:
            ohs_prompt_text (str): The OHS prompt string.

        Returns:
            tuple: Contains lists of true_counts, pred_counts, true_risks, pred_risks, 
                   and all_predictions_for_file (raw dicts for saving).
        """
        self.logger.info("Collecting predictions and ground truths...")
        if not self.model or not self.processor:
            self.logger.error("Model not loaded. Call load_model_for_evaluation() first.")
            raise ValueError("Model not loaded.")
        if not self.test_examples:
            self.logger.warning("No test examples available to evaluate. Returning empty results.")
            return [], [], [], [], []

        pred_counts, true_counts = [], []
        pred_risks, true_risks = [], []
        all_predictions_for_file = []

        for samp in tqdm(self.test_examples, desc="Evaluating samples"):
            video_path_str = samp.get('video_path', 'Unknown Video')
            gt_text = samp.get("messages", [{}, {}])[1].get("content", [{}])[0].get("text", "{}")
            
            gt_json = {}
            try:
                match_gt = re.search(r"\{[\s\S]*\}", gt_text)
                if match_gt: gt_json = json.loads(match_gt.group())
            except json.JSONDecodeError: 
                self.logger.warning(f"Could not parse ground truth JSON for {video_path_str}. GT text: {gt_text}")
            
            pred_json = self._generate_reply_for_sample(samp, ohs_prompt_text) # Already handles its own errors
            
            all_predictions_for_file.append({"video_path": video_path_str, "ground_truth": gt_json, "prediction": pred_json})
            
            true_obj_counts = gt_json.get("object_counts", {})
            pred_obj_counts = pred_json.get("object_counts", {})
            true_counts.append(tuple(true_obj_counts.get(k, 0) for k in self.metric_obj_keys))
            pred_counts.append(tuple(pred_obj_counts.get(k, 0) for k in self.metric_obj_keys))
            
            true_detected_risks = gt_json.get("detected_risks", [])
            pred_detected_risks = pred_json.get("detected_risks", [])
            # Ensure risk numbers are extracted correctly and handle potential None values
            true_risks.append({r.get("risk_number") for r in true_detected_risks if r and isinstance(r, dict) and "risk_number" in r})
            pred_risks.append({r.get("risk_number", -1) for r in pred_detected_risks if r and isinstance(r, dict) and "risk_number" in r})
        
        self.logger.info("Inference and ground truth collection complete.")
        return true_counts, pred_counts, true_risks, pred_risks, all_predictions_for_file

    def calculate_and_save_metrics(self, true_counts, pred_counts, true_risks, pred_risks, all_predictions_for_file) -> None:
        """
        Calculates object count metrics (MAE, RMSE, R2) and risk detection metrics 
        (micro-P/R/F1). Saves raw predictions and a metrics summary to files.
        
        Args:
            true_counts (list): List of tuples of true object counts.
            pred_counts (list): List of tuples of predicted object counts.
            true_risks (list): List of sets of true risk numbers.
            pred_risks (list): List of sets of predicted risk numbers.
            all_predictions_for_file (list): List of dicts containing raw GT and predictions.
        """
        self.logger.info("Calculating and saving metrics...")
        eval_results_dir = self.exp_manager.get_evaluation_results_dir()
        predictions_file_path = eval_results_dir / "predictions.json"
        try:
            with open(predictions_file_path, 'w', encoding='utf-8') as f: 
                json.dump(all_predictions_for_file, f, indent=4)
            self.logger.info(f"Raw predictions saved to {predictions_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save raw predictions to {predictions_file_path}: {e}")

        metrics_summary_lines = [f"Evaluation Metrics for Experiment: {self.exp_manager.current_experiment_path.name}"]
        
        # Object Count Metrics
        metrics_summary_lines.append("\nObject Count Metrics:")
        msg_counts = "Not enough valid data for object count metrics."
        if true_counts and pred_counts and len(true_counts) == len(pred_counts) and len(true_counts) > 0:
            try:
                true_arr, pred_arr = np.array(true_counts), np.array(pred_counts)
                if true_arr.ndim == 2 and pred_arr.ndim == 2 and true_arr.shape == pred_arr.shape and true_arr.shape[0] >= 2 : # R2 needs at least 2 samples
                    mae = mean_absolute_error(true_arr, pred_arr, multioutput='raw_values')
                    mse = mean_squared_error(true_arr, pred_arr, multioutput="raw_values")
                    rmse = np.sqrt(mse)
                    r2 = r2_score(true_arr, pred_arr, multioutput='raw_values')
                    # Ensure metric_obj_keys match the number of columns in true_arr/pred_arr
                    columns_for_df = self.metric_obj_keys
                    if len(self.metric_obj_keys) != true_arr.shape[1]:
                        self.logger.warning(f"Mismatch between OBJECT_KEYS_FOR_METRICS ({len(self.metric_obj_keys)}) and data columns ({true_arr.shape[1]}). Using generic column names.")
                        columns_for_df = [f"obj_{i+1}" for i in range(true_arr.shape[1])]
                    
                    df = pd.DataFrame([mae, rmse, r2], index=["MAE","RMSE", "R2"], columns=columns_for_df)
                    msg_counts = df.round(3).to_string()
                    self.logger.info(f"\nObject Count Metrics calculated:\n{msg_counts}")
                else:
                    self.logger.warning(f"Could not calculate object count metrics. Conditions not met. True samples: {len(true_arr)}, Pred samples: {len(pred_arr)}, Min samples for R2: 2. Array shapes: true={true_arr.shape}, pred={pred_arr.shape}")
            except Exception as e:
                self.logger.error(f"Error calculating object count metrics: {e}", exc_info=True)
        else:
            self.logger.warning(msg_counts + f" (True counts: {len(true_counts)}, Pred counts: {len(pred_counts)})")
        metrics_summary_lines.append(msg_counts)

        # Risk Detection Metrics
        metrics_summary_lines.append(f"\nRisk Micro-Precision / Recall / F1-score:")
        msg_risks = "Not enough valid data for risk detection P/R/F1 metrics."
        if true_risks and pred_risks and len(true_risks) == len(pred_risks) and len(true_risks) > 0:
            try:
                # Filter out -1 (used as a placeholder for no prediction) before creating all_labels
                valid_true_risks = [s - {-1} for s in true_risks]
                valid_pred_risks = [s - {-1} for s in pred_risks]

                all_labels = sorted(list(set.union(*valid_true_risks, *valid_pred_risks)))
                if all_labels: # Only proceed if there are actual risk labels
                    y_true_ml = [[int(lbl in s) for lbl in all_labels] for s in valid_true_risks]
                    y_pred_ml = [[int(lbl in s) for lbl in all_labels] for s in valid_pred_risks]
                    
                    # Check if there's any activity (at least one true positive, false positive, or false negative)
                    if np.any(y_true_ml) or np.any(y_pred_ml):
                        prf = precision_recall_fscore_support(y_true_ml, y_pred_ml, average='micro', zero_division=0)
                        msg_risks = f"Precision: {prf[0]:.3f} / Recall: {prf[1]:.3f} / F1-score: {prf[2]:.3f} (micro average)"
                        self.logger.info(f"\nRisk Detection Metrics: {msg_risks}")
                    else:
                        msg_risks = "No positive risk labels identified in ground truth or predictions after filtering. P/R/F1 are undefined or 0."
                        self.logger.warning(msg_risks)
                else:
                    msg_risks = "No valid risk labels (other than -1) found after processing. Cannot calculate P/R/F1."
                    self.logger.warning(msg_risks)
            except Exception as e:
                self.logger.error(f"Error calculating risk detection metrics: {e}", exc_info=True)
        else:
             self.logger.warning(msg_risks + f" (True risks: {len(true_risks)}, Pred risks: {len(pred_risks)})")
        metrics_summary_lines.append(msg_risks)
        
        metrics_file_path = eval_results_dir / "evaluation_metrics.txt"
        try:
            with open(metrics_file_path, 'w', encoding='utf-8') as f: 
                f.write("\n".join(metrics_summary_lines))
            self.logger.info(f"Evaluation metrics summary saved to {metrics_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation metrics to {metrics_file_path}: {e}")
            
        self.logger.info(f"All evaluation artifacts for this run are in: {eval_results_dir}")

    def run_evaluation_pipeline(self) -> None:
        """
        Executes the full evaluation pipeline: model loading, data preparation,
        prediction collection, and metrics calculation/saving.
        
        Raises:
            Exception: Propagates any exception that occurs after logging it.
        """
        self.logger.info(f"Starting full evaluation pipeline for experiment: {self.exp_manager.current_experiment_path}")
        try:
            self.load_model_for_evaluation() # Exits on critical model load failure
            
            ohs_prompt_path_str = self.exp_manager.get_config_value("OHS_PROMPT_PATH", main_config_module=self.global_config)
            if not ohs_prompt_path_str:
                self.logger.critical("OHS_PROMPT_PATH not found in config. Cannot proceed."); sys.exit(1)
            
            ohs_prompt_path = pathlib.Path(ohs_prompt_path_str)
            if not ohs_prompt_path.exists():
                 self.logger.critical(f"OHS prompt file not found at {ohs_prompt_path}. Cannot proceed."); sys.exit(1)
            
            ohs_prompt_text = ohs_prompt_path.read_text(encoding="utf-8").strip()
            self.logger.info(f"Loaded OHS prompt from {ohs_prompt_path}")
            
            self.prepare_evaluation_data(ohs_prompt_text) # Exits on critical data prep failure
            if not self.test_examples:
                self.logger.warning("No test examples were prepared. Evaluation pipeline cannot continue.")
                return

            true_counts, pred_counts, true_risks, pred_risks, all_preds_file = self.collect_predictions_and_ground_truths(ohs_prompt_text)
            
            self.calculate_and_save_metrics(true_counts, pred_counts, true_risks, pred_risks, all_preds_file)
            self.logger.info("Evaluation pipeline finished successfully.")
        except SystemExit: # Allow sys.exit to propagate to stop execution
            raise   
        except Exception as e:
            self.logger.exception(f"Critical error during evaluation pipeline for experiment {self.exp_manager.current_experiment_path}. Details: {e}")
            raise 
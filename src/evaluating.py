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
        test_ordered_video_names (list): List to maintain order of test set.
        test_ground_truth_map (dict): Dictionary to store ground truth answers keyed by video_name.
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
        self.test_ordered_video_names = [] # To maintain order of test set
        self.test_ground_truth_map = {} # To store GT answers keyed by video_name
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

    def _extract_text_prompt(self, sample: dict) -> str | None:
        """Extracts the text prompt from a sample dictionary."""
        self.logger.debug(f"Extracting text prompt for sample related to video: {sample.get('video_path', 'Unknown')}")
        user_content = sample.get("messages", [{}])[0].get("content", [])
        for blk in user_content:
            if blk.get("type") == "text":
                return blk.get("text")
        self.logger.warning(f"No text prompt found in sample {sample.get('video_path', 'Unknown')}")
        return None

    def _generate_reply_for_sample(self, sample: dict) -> dict:
        """
        Generates a model reply (parsed JSON) for a single evaluation sample.
        
        Args:
            sample (dict): The input sample dictionary, expected to be in the format 
                           returned by row_to_example (i.e., {"messages": [...]}).

        Returns:
            dict: Parsed JSON from the model's reply, or an empty dict on failure.
        """
        video_p_from_messages = "Unknown" # Placeholder if video_path not in messages (it shouldn't be)
                                        # Actual video_path for logging should be retrieved before calling this if needed.
        
        # sample is now directly the output of row_to_example, which is {"messages": [...]}
        # The prompt and images are within sample["messages"][0]["content"]
        
        # For logging purposes, it's better if video_path was part of the sample when iterating
        # However, the current `self.test_examples` only contains `{"messages": ...}`.
        # Let's assume for now `video_p_from_messages` is sufficient if we can't easily get original video_path here.

        self.logger.debug(f"Generating reply for sample (structure: {sample.keys()})") # Log keys of sample
        
        images = self._extract_images(sample) # sample is {"messages": ...}
        if not images: 
            self.logger.warning(f"No images found or loadable for sample, cannot generate reply.")
            return {}

        ohs_prompt_from_sample = self._extract_text_prompt(sample) # sample is {"messages": ...}
        if not ohs_prompt_from_sample:
            self.logger.warning(f"No OHS prompt extracted from sample, cannot generate reply.")
            return {}
        
        # messages_for_model should be directly sample["messages"] if it's for the user role
        # and doesn't contain an assistant response.
        # The processor.apply_chat_template usually handles the full conversation.
        # Let's ensure the 'sample' is structured as the input for apply_chat_template.
        # row_to_example creates: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        # For inference, we only need the user part.
        
        user_messages_for_inference = [msg for msg in sample["messages"] if msg.get("role") == "user"]
        if not user_messages_for_inference:
            self.logger.warning("No user messages found in sample for inference.")
            return {}

        try:
            # The processor.apply_chat_template expects the full conversation usually,
            # but for generation, providing only the user turn(s) and add_generation_prompt=True is common.
            text = self.processor.apply_chat_template(user_messages_for_inference, tokenize=False, add_generation_prompt=True)
            
            # process_vision_info might also expect the "messages" structure
            # It needs to extract images from user_messages_for_inference
            image_inputs, video_inputs = process_vision_info(user_messages_for_inference) 
            
            batch = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device_cfg)
            
            with torch.inference_mode():
                gen_ids = self.model.generate(**batch, **self.eval_gen_cfg, use_cache=True)
            
            gen_trim = gen_ids[:, batch.input_ids.shape[1]:]
            reply = self.processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            self.logger.debug(f"Generated reply for {video_p_from_messages} (first 100 chars): {reply[:100]}...")
            
            match = re.search(r"\{[\s\S]*\}", reply)
            if match:
                parsed_json = json.loads(match.group())
                return parsed_json
            else:
                self.logger.warning(f"No JSON-like structure found in reply for {video_p_from_messages}. Reply: {reply}")
                return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError parsing reply for {video_p_from_messages}. Reply: {reply}. Error: {e}")
            return {}
        except Exception as e:
            self.logger.exception(f"Unexpected error during reply generation for {video_p_from_messages}: {e}")
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
        filters the main CSV for these videos, creates example dicts, and stores
        ground truth information for later use.
        Sets `self.test_examples`, `self.test_ordered_video_names`, `self.test_ground_truth_map`.
        Exits if critical files are missing.
        
        Args:
            ohs_prompt_text (str): The OHS prompt string.
        """
        self.logger.info("Preparing evaluation data...")
        
        # Determine path to test video names file from the experiment directory
        data_split_dir = self.exp_manager.get_data_split_dir(self.global_config.DATA_SPLIT_DIR_NAME)
        test_video_names_path = data_split_dir / "test_video_names.txt"
        
        # Load the test video names, which are ordered and will be used to create test_examples
        self.test_ordered_video_names = self._load_video_names(test_video_names_path)
        if not self.test_ordered_video_names:
            self.logger.critical(f"No test video names found in {test_video_names_path}. Cannot proceed with evaluation.")
            sys.exit(1)

        # Load the original CSV to get ground truth answers
        try:
            ground_truth_df = pd.read_csv(
                self.global_config.CSV_PATH, 
                header=None, 
                names=["video_name", "gemini_answer"],
                encoding="utf-8"
            )
            # Create a dictionary for quick lookup of ground truth answers.
            # Key: video_name (string), Value: gemini_answer (string)
            self.test_ground_truth_map = pd.Series(
                ground_truth_df.gemini_answer.values, 
                index=ground_truth_df.video_name
            ).to_dict()
            self.logger.info(f"Loaded {len(self.test_ground_truth_map)} ground truth answers from {self.global_config.CSV_PATH}")
        except FileNotFoundError:
            self.logger.critical(f"Ground truth CSV file not found at {self.global_config.CSV_PATH}. Cannot proceed.")
            sys.exit(1)

        # Prepare test examples for the model
        self.test_examples = []
        for video_name_with_ext in self.test_ordered_video_names:
            # Bug Fix: The CSV has video names *without* extension.
            # We must remove the extension before using it as a key for the ground truth map.
            video_name_no_ext = pathlib.Path(video_name_with_ext).stem
            
            # Construct a dummy row for row_to_example, which needs 'video_name' and 'gemini_answer'
            # The 'gemini_answer' here is the ground truth, used by row_to_example to formulate the assistant's turn.
            # For inference, this part of the example will be stripped out before sending to the model,
            # but it is needed to create the example structure consistently.
            dummy_row = pd.Series({
                "video_name": video_name_with_ext,
                "gemini_answer": self.test_ground_truth_map.get(video_name_no_ext, "") # Use name without extension
            })
            
            # The row_to_example function creates the structured message for the model
            model_input_data = row_to_example(dummy_row, ohs_prompt_text)
            self.test_examples.append(model_input_data)
        
        self.logger.info(f"Prepared {len(self.test_examples)} examples for evaluation from the test set.")

    def collect_predictions_and_ground_truths(self) -> tuple[list, list, list, list, list]:
        """
        Iterates through `self.test_examples`, generates model predictions,
        and extracts ground truth information using pre-populated attributes 
        `self.test_ordered_video_names` and `self.test_ground_truth_map`.
        
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
        if len(self.test_examples) != len(self.test_ordered_video_names):
            self.logger.error(f"Mismatch between test_examples ({len(self.test_examples)}) and ordered video names ({len(self.test_ordered_video_names)}). Data integrity issue.")
            # Depending on severity, could raise an error or return empty
            return [], [], [], [], [] # Return empty to prevent further processing with mismatched data

        true_counts, pred_counts = [], []
        true_risks, pred_risks = [], []
        all_predictions_for_file = []

        for idx, sample in enumerate(tqdm(self.test_examples, desc="Generating predictions")):
            video_name_with_ext = self.test_ordered_video_names[idx]
            self.logger.debug(f"Processing for evaluation: {video_name_with_ext}")

            pred_json = self._generate_reply_for_sample(sample)
            
            # Bug Fix: The map key is the video name *without* the extension.
            video_name_no_ext = pathlib.Path(video_name_with_ext).stem
            gt_json_str = self.test_ground_truth_map.get(video_name_no_ext)
            parsed_gt_json = {}
            if gt_json_str:
                try:
                    if isinstance(gt_json_str, str):
                        parsed_gt_json = json.loads(gt_json_str)
                    elif isinstance(gt_json_str, dict): # Should not happen if from CSV
                        parsed_gt_json = gt_json_str
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse ground truth JSON for {video_name_with_ext}. GT text: {gt_json_str}")
            else:
                self.logger.warning(f"No ground truth data found in map for video: {video_name_with_ext}")
            
            current_prediction_record = {
                "video_name": video_name_with_ext, 
                "prediction": pred_json if pred_json else {},
                "ground_truth": parsed_gt_json
            }
            all_predictions_for_file.append(current_prediction_record)

            if not pred_json: 
                self.logger.warning(f"Skipping metrics for {video_name_with_ext} due to empty/failed prediction.")
                true_counts.append(tuple(0 for _ in self.metric_obj_keys))
                pred_counts.append(tuple(0 for _ in self.metric_obj_keys))
                true_risks.append([]) 
                pred_risks.append([])
                continue

            true_obj_counts_dict = parsed_gt_json.get("object_counts", {})
            pred_obj_counts_dict = pred_json.get("object_counts", {})
            true_counts.append(tuple(true_obj_counts_dict.get(k, 0) for k in self.metric_obj_keys))
            pred_counts.append(tuple(pred_obj_counts_dict.get(k, 0) for k in self.metric_obj_keys))
            
            true_detected_risks_list = parsed_gt_json.get("detected_risks", [])
            pred_detected_risks_list = pred_json.get("detected_risks", [])
            true_risks.append(sorted([str(r).lower() for r in true_detected_risks_list]))
            pred_risks.append(sorted([str(r).lower() for r in pred_detected_risks_list]))
        
        self.logger.info(f"Collected {len(all_predictions_for_file)} predictions/ground truths.")
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
        Orchestrates the full evaluation pipeline: loads config, model, data,
        collects predictions, calculates metrics, and saves results.
        """
        self.logger.info("Starting evaluation pipeline...")
        
        ohs_prompt_path = pathlib.Path(self.exp_manager.get_config_value("OHS_PROMPT_PATH", main_config_module=self.global_config))
        try:
            with open(ohs_prompt_path, "r", encoding="utf-8") as f:
                ohs_prompt_text = f.read().strip()
            self.logger.info(f"Loaded OHS prompt for evaluation from: {ohs_prompt_path}")
        except Exception as e:
            self.logger.critical(f"Failed to load OHS prompt from {ohs_prompt_path}: {e}"); sys.exit(1)

        self.load_model_for_evaluation()
        self.prepare_evaluation_data(ohs_prompt_text) 

        if not self.test_examples:
            self.logger.warning("No test examples were prepared. Evaluation pipeline cannot continue meaningfully.")
            # Optionally save an empty metrics file or just return
            self.calculate_and_save_metrics([], [], [], [], []) # Save empty/default metrics
            return

        true_counts, pred_counts, true_risks, pred_risks, all_preds_for_file = self.collect_predictions_and_ground_truths() # Removed ohs_prompt_text
        
        self.calculate_and_save_metrics(true_counts, pred_counts, true_risks, pred_risks, all_preds_for_file)
        self.logger.info("Evaluation pipeline finished successfully.") 
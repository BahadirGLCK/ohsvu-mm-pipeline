import sys, json, pathlib, re, io, random # random for seed only
# import subprocess, importlib.metadata, uuid # Not used
# import torch, evaluate, numpy as np, pandas as pd # Moved to Evaluator or no longer direct use
# from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score # Moved to Evaluator
# from transformers import AutoProcessor # Moved to Evaluator
# from unsloth import FastVisionModel # Moved to Evaluator
# from peft import PeftModel # Moved to Evaluator
# from qwen_vl_utils import process_vision_info # Moved to Evaluator
# from PIL import Image # Moved to Evaluator
# from io import BytesIO # Moved to Evaluator
# from tqdm import tqdm # Moved to Evaluator
import logging

# from .utils import row_to_example # Moved to Evaluator
from . import config as global_config 
from .experiment_manager import ExperimentManager
from .evaluating import Evaluator

# Global seed for this module, if any top-level random operations were needed.
# Evaluator class handles its own specific seeding if necessary via config.
random.seed(global_config.EVAL_RANDOM_SEED_SAMPLING)

# Module-level logger, primarily for main_evaluation_process orchestration messages
# and the __main__ block. Evaluator and ExperimentManager have their own loggers.
logger = logging.getLogger(__name__)

# The helper functions (_load_video_names, _extract_images, _generate_reply_for_sample,
# _prepare_evaluation_data, _collect_predictions_and_ground_truths, 
# _calculate_and_save_metrics) have been moved into the Evaluator class (src/evaluating.py).
# They are removed from here to centralize evaluation logic within the Evaluator class.

def main_evaluation_process(experiment_dir_path: pathlib.Path) -> None:
    """
    Orchestrates the model evaluation process for a given experiment directory.

    This function initializes an ExperimentManager to target the specified experiment 
    directory. It then instantiates and runs the Evaluator's pipeline, which handles
    model loading, data preparation, inference, and metric calculation/saving.

    Args:
        experiment_dir_path (pathlib.Path): The path to the specific experiment 
                                            directory that contains the finetuned model
                                            and configuration to be evaluated.
    """
    
    # Initialize ExperimentManager. This will also set up its own logger for the specified experiment.
    # Pass the root dir from global_config for context, but it will primarily operate on experiment_dir_path.
    exp_manager = ExperimentManager(
        experiments_root_dir=global_config.EXPERIMENT_ROOT_DIR, 
        experiment_name_prefix="evaluate" # Default prefix, but path is being set directly
    )
    
    try:
        # Set the ExperimentManager to operate on the provided experiment_dir_path.
        # This also loads its config and sets up the experiment-specific logger.
        exp_manager.set_current_experiment_path(experiment_dir_path)
    except FileNotFoundError as e:
        # If ExperimentManager fails to set the path (e.g., dir not found),
        # log critical error and exit. The exp_manager.logger might not be fully set up.
        logger.critical(f"Target experiment directory not found: {experiment_dir_path}. Error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error setting up ExperimentManager for {experiment_dir_path}: {e}", exc_info=True)
        sys.exit(1)

    # Use the logger from exp_manager as it's now configured for the specific experiment.
    effective_logger = exp_manager.logger if exp_manager.logger else logger

    effective_logger.info(f"Starting evaluation process for experiment: {experiment_dir_path}")
    
    # Instantiate the Evaluator, passing the global config and the configured ExperimentManager.
    # The Evaluator will use the experiment_dir_path via the exp_manager.
    evaluator_instance = Evaluator(global_config_module=global_config, experiment_manager=exp_manager)

    try:
        # The Evaluator's pipeline handles all evaluation steps.
        evaluator_instance.run_evaluation_pipeline()
        effective_logger.info(f"Evaluation pipeline completed successfully for experiment: {experiment_dir_path}")
    except SystemExit:
        effective_logger.warning(f"Evaluation pipeline for {experiment_dir_path} exited prematurely (SysExit).")
        raise # Re-raise SystemExit to ensure script termination
    except Exception as e:
        # The Evaluator.run_evaluation_pipeline should have logged detailed errors.
        effective_logger.error(
            f"Evaluation pipeline encountered an error for experiment {experiment_dir_path}. Error: {e}", 
            exc_info=True # Adds traceback information
        )
        effective_logger.info(f"Please check logs in the experiment directory: {experiment_dir_path / 'experiment_run.log'} for details.")
        # Depending on desired behavior, could re-raise 'e' or handle differently.

if __name__ == "__main__":
    # Basic logging configuration for direct script execution.
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] # Ensure console output for direct runs
    )
    logger.info("Running evaluation process directly via __main__...")

    if len(sys.argv) > 1:
        exp_path_arg = pathlib.Path(sys.argv[1]).resolve() # Resolve path immediately
        logger.info(f"Attempting to evaluate experiment at: {exp_path_arg}")
        
        # Basic check, ExperimentManager will do a more thorough one in set_current_experiment_path
        if not exp_path_arg.is_dir(): 
            logger.critical(f"Error: Provided experiment path '{exp_path_arg}' is not a valid directory. Exiting.")
            sys.exit(1)
        try:
            main_evaluation_process(exp_path_arg)
            logger.info(f"Evaluation process (direct run) concluded for experiment: {exp_path_arg}")
        except SystemExit:
            logger.info(f"Direct run of evaluation for {exp_path_arg} exited as expected.")
        except Exception as e:
            logger.critical(f"A critical error occurred during the direct run of the evaluation process for {exp_path_arg}.", exc_info=True)
    else: 
        logger.info("Usage: python -m src.evaluation <path_to_experiment_directory>")
        logger.info("Please provide a valid experiment directory path as a command-line argument.")
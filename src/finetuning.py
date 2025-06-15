import os, json, math, random, pathlib, typing as T
# from dataclasses import dataclass # Not used
# from datetime import datetime # Not used

# from tqdm import tqdm # No longer used directly here, Trainer uses it
# from PIL import Image # No longer used directly here
# import pandas as pd # No longer used directly here
# from datasets import Dataset # No longer used directly here

# These imports are also primarily for Trainer, but kept here if any _functions were to be reused standalone with type hints
# from unsloth import FastVisionModel 
# from unsloth.trainer import (
#     UnslothVisionDataCollator,
#     UnslothTrainer,
#     UnslothTrainingArguments,
# )
# from transformers import AutoProcessor

# from .utils import row_to_example # No longer used directly here
from . import config as global_config
from .experiment_manager import ExperimentManager
from .training import Trainer

import logging

# Module-level random seed, consistent with Trainer's initialization
random.seed(global_config.RANDOM_SEED)

# Logger for this module, primarily for main_finetune_process orchestration messages
# and the __main__ block. Trainer and ExperimentManager have their own loggers.
logger = logging.getLogger(__name__)

# The _prepare_data_splits, _save_data_split_info, _initialize_and_prepare_model,
# _train_model, and _save_model_artifacts functions have been moved into the 
# Trainer class (src/training.py). They are removed from here to avoid duplication
# and to centralize training logic within the Trainer class.

def main_finetune_process(continue_experiment: str | None = None, checkpoint_path: str | None = None) -> pathlib.Path:
    """
    Orchestrates the model finetuning process.

    This function either initializes a new experiment or continues an existing one,
    saves a snapshot of the global configuration, and then instantiates and runs
    the Trainer's training pipeline.

    Args:
        continue_experiment (str | None): Path to an existing experiment directory to continue training from.
                                         If None, a new experiment will be created.
        checkpoint_path (str | None): Path to a specific checkpoint within the experiment directory to resume from.
                                     If None, will use the latest checkpoint in the experiment directory.

    Returns:
        pathlib.Path: The path to the experiment directory where artifacts and logs 
                      for this finetuning run are stored. If an error occurs, it might
                      return the path for inspection of partial results.
    """
    
    if continue_experiment:
        # Use existing experiment directory
        current_experiment_path = pathlib.Path(continue_experiment).resolve()
        if not current_experiment_path.is_dir():
            raise ValueError(f"Experiment directory not found: {continue_experiment}")
        
        exp_manager = ExperimentManager(
            global_config.EXPERIMENT_ROOT_DIR,
            experiment_name_prefix="finetune"
        )
        exp_manager.current_experiment_path = current_experiment_path
        exp_manager._setup_experiment_logger()  # Initialize logger for the existing experiment
    else:
        # Create new experiment
        exp_manager = ExperimentManager(
            global_config.EXPERIMENT_ROOT_DIR, 
            experiment_name_prefix="finetune"
        )
        current_experiment_path = exp_manager.create_new_experiment()
    
    # Use the logger from ExperimentManager as it's configured for the specific experiment
    effective_logger = exp_manager.logger if exp_manager.logger else logger

    if not continue_experiment:
        effective_logger.info(f"Saving global configuration snapshot for experiment: {current_experiment_path}")
        exp_manager.save_config_snapshot(global_config)

    effective_logger.info(f"Starting finetuning process for experiment: {current_experiment_path}")
    if checkpoint_path:
        effective_logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    elif continue_experiment:
        effective_logger.info("Continuing from latest checkpoint in experiment directory")

    # Instantiate the Trainer, passing the global config and the initialized ExperimentManager
    trainer_instance = Trainer(config=global_config, experiment_manager=exp_manager)
    
    try:
        # The Trainer's pipeline handles all steps: data prep, model init, training, saving
        trained_experiment_path = trainer_instance.run_training_pipeline(
            continue_experiment=continue_experiment,
            checkpoint_path=checkpoint_path
        )
        effective_logger.info(f"Finetuning pipeline completed successfully. Artifacts in: {trained_experiment_path}")
        return trained_experiment_path
    except Exception as e:
        # The Trainer.run_training_pipeline should have logged detailed errors.
        # This log provides a high-level notification that the process in main_finetune_process failed.
        effective_logger.error(
            f"Finetuning pipeline encountered an error for experiment {current_experiment_path}. Error: {e}", 
            exc_info=True # Adds traceback information to the log
        )
        effective_logger.info(f"Please check the logs in the experiment directory: {current_experiment_path / 'experiment_run.log'} for detailed error information.")
        # Return the path for inspection of any partial artifacts or logs
        return current_experiment_path 

if __name__ == "__main__":
    # Basic logging configuration for direct script execution.
    # This ensures that messages are displayed if the script is run as the main module,
    # especially before the ExperimentManager's specific logger is initialized.
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] # Ensure logs go to console
    )
    logger.info("Running finetuning process directly via __main__...")
    try:
        experiment_output_path = main_finetune_process()
        logger.info(f"Finetuning process (direct run) concluded. Experiment path: {experiment_output_path}")
    except Exception as e:
        # This will catch exceptions not handled within main_finetune_process or its calls,
        # though most should be caught and logged there.
        logger.critical("A critical error occurred during the direct execution of the finetuning process.", exc_info=True)

import sys
import os
import pathlib
import argparse
import logging

# Setup basic logging for the main script execution itself.
# This will catch logs before ExperimentManager specific logger is active 
# or if stages are skipped. Configures a stream handler for console output.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Explicitly stream to stdout
)
logger = logging.getLogger(__name__) # Logger for the main.py orchestration scope

# Ensure the project root is in PYTHONPATH to allow `from src...` imports
# This assumes main.py is in the project root directory.
project_root = pathlib.Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the main process functions from the src package
from src.finetuning import main_finetune_process
from src import evaluation # Keep import style for hasattr compatibility

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Runs the OHSVU pipeline, which can include finetuning 
    and evaluation stages based on the provided command-line arguments.

    The pipeline manages experiment directories and allows specifying an existing
    experiment for evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments controlling 
                                   pipeline execution.
                                   Expected attributes:
                                   - skip_finetuning (bool)
                                   - skip_evaluation (bool)
                                   - eval_experiment_dir (str | None)
                                   - checkpoint_path (str | None)
    """
    logger.info("Starting the OHSVU pipeline...")
    logger.debug(f"Pipeline called with arguments: {args}")
    
    experiment_dir_from_finetuning: pathlib.Path | None = None

    # --- Step 1: Finetuning (Optional) ---
    if not args.skip_finetuning:
        logger.info("--- Initiating Step 1: Finetuning ---")
        try:
            # main_finetune_process creates a new experiment dir and returns its path.
            # It uses ExperimentManager, which sets up its own experiment-specific logger.
            experiment_dir_from_finetuning = main_finetune_process(checkpoint_path=args.checkpoint_path)
            if experiment_dir_from_finetuning and experiment_dir_from_finetuning.is_dir():
                logger.info(f"Finetuning completed. Artifacts and logs in: {experiment_dir_from_finetuning}")
                logger.info(f"Check main log: {experiment_dir_from_finetuning / 'experiment_run.log'}")
            else:
                logger.warning("Finetuning process finished but did not return a valid experiment directory path. Evaluation might be affected.")
        except Exception as e:
            logger.error(f"A critical error occurred during the finetuning process: {e}", exc_info=True)
            experiment_dir_from_finetuning = None # Ensure it's None if finetuning failed
            logger.info("Finetuning step failed. Proceeding according to pipeline arguments.")
    else:
        logger.info("--- Skipping Finetuning step as per --skip_finetuning argument. ---")

    # --- Step 2: Evaluation (Optional) ---
    if not args.skip_evaluation:
        logger.info("--- Initiating Step 2: Evaluation ---")
        
        experiment_to_evaluate: pathlib.Path | None = None

        # Determine the target experiment directory for evaluation:
        # Priority 1: Explicitly provided --eval_experiment_dir
        # Priority 2: Directory created by the finetuning step (if run and successful)
        if args.eval_experiment_dir:
            eval_dir_path = pathlib.Path(args.eval_experiment_dir).resolve()
            if eval_dir_path.is_dir():
                logger.info(f"Using user-specified experiment directory for evaluation: {eval_dir_path}")
                experiment_to_evaluate = eval_dir_path
            else:
                logger.error(f"The --eval_experiment_dir '{args.eval_experiment_dir}' was not found or is not a directory. Skipping evaluation.")
        elif experiment_dir_from_finetuning and experiment_dir_from_finetuning.is_dir():
            logger.info(f"Using experiment directory from the preceding finetuning step for evaluation: {experiment_dir_from_finetuning}")
            experiment_to_evaluate = experiment_dir_from_finetuning
        else:
            logger.warning("Evaluation step requires an experiment directory. None was provided via --eval_experiment_dir, and no valid directory resulted from finetuning (if run). Skipping evaluation.")

        if experiment_to_evaluate: # Proceed only if a valid directory path is determined
            try:
                # Guard the call to main_evaluation_process from src.evaluation
                if hasattr(evaluation, "main_evaluation_process") and callable(evaluation.main_evaluation_process):
                    # main_evaluation_process uses ExperimentManager for its specific experiment dir and logger.
                    evaluation.main_evaluation_process(experiment_to_evaluate)
                    logger.info(f"Evaluation completed for experiment: {experiment_to_evaluate}")
                    logger.info(f"Check main log: {experiment_to_evaluate / 'experiment_run.log'}")
                else:
                    logger.critical("The function src.evaluation.main_evaluation_process() was not found or is not callable. This is an unexpected error.")
            except SystemExit:
                logger.warning(f"Evaluation for {experiment_to_evaluate} exited prematurely via SystemExit.") # Usually from a sys.exit() call within
            except Exception as e:
                logger.error(f"A critical error occurred during the evaluation process for {experiment_to_evaluate}: {e}", exc_info=True)
        elif not args.eval_experiment_dir: # Log skip only if not due to a bad --eval_experiment_dir (which logs its own error)
            logger.info("Skipping evaluation as no valid experiment directory was determined.")
    else:
        logger.info("--- Skipping Evaluation step as per --skip_evaluation argument. ---")

    logger.info("OHSVU pipeline execution finished.")

if __name__ == "__main__":
    # Command-line argument parsing setup
    parser = argparse.ArgumentParser(
        description="OHSVU Finetuning and Evaluation Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Provides default values in help messages
    )
    parser.add_argument(
        "--skip_finetuning", 
        action="store_true", 
        help="If set, the finetuning process will be skipped."
    )
    parser.add_argument(
        "--skip_evaluation", 
        action="store_true", 
        help="If set, the evaluation process will be skipped."
    )
    parser.add_argument(
        "--eval_experiment_dir", 
        type=str, 
        default=None, # No default, evaluation path must be explicit or from finetuning
        help="Path to a specific, pre-existing experiment directory to be used for evaluation. "
             "If finetuning is run, its output directory is used for evaluation unless this is set."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory to resume training from. "
             "If not provided, training will start from scratch."
    )
    
    parsed_args = parser.parse_args()
    
    logger.info(f"OHSVU pipeline initiated with arguments: {parsed_args}")
    run_pipeline(parsed_args) 
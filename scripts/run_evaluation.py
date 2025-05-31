import sys
import os
import argparse # For command-line arguments
import pathlib

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation import main_evaluation_process # Import the evaluation logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a fine-tuned model from a specific experiment directory.")
    parser.add_argument(
        "experiment_dir", 
        type=str, 
        help="Path to the timestamped experiment directory containing the fine-tuned model and config."
    )
    args = parser.parse_args()

    experiment_path = pathlib.Path(args.experiment_dir).resolve() # Resolve to absolute path

    if not experiment_path.is_dir():
        print(f"Error: Experiment directory not found: {experiment_path}")
        sys.exit(1)

    print(f"Starting evaluation process for experiment: {experiment_path}")
    try:
        main_evaluation_process(experiment_path)
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Ensure that the 'src' directory is in the Python path and all dependencies are installed.")
    except Exception as e:
        print(f"An error occurred during the evaluation process for {experiment_path}: {e}")
        # raise
    finally:
        print(f"Evaluation process script finished for experiment: {experiment_path}") 
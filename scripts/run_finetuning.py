import sys
import os

# Add the project root to the Python path to allow importing from src
# This assumes the script is in project_root/scripts/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.finetuning import main_finetune_process

if __name__ == "__main__":
    print("Starting finetuning process...")
    try:
        experiment_path = main_finetune_process()
        print(f"Finetuning process completed. Experiment artifacts are in: {experiment_path}")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Ensure that the 'src' directory is in the Python path and all dependencies are installed.")
    except Exception as e:
        print(f"An error occurred during the finetuning process: {e}")
        # It might be helpful to re-raise the exception if debugging is needed or for more detailed logs
        # raise
    finally:
        print("Finetuning process script finished.") 
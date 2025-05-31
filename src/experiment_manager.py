import pathlib
import datetime
import json
import logging

class ExperimentManager:
    """
    Manages all aspects of a single experiment run, including directory creation,
    configuration snapshotting, path provision for artifacts, and experiment-specific logging.

    Attributes:
        experiments_root_dir (pathlib.Path): The root directory where all experiment 
                                             subdirectories will be created.
        experiment_name_prefix (str): A prefix string for naming experiment directories.
        logger (logging.Logger | None): The logger instance for this experiment.
                                        Initialized by _setup_experiment_logger().
        _current_experiment_path (pathlib.Path | None): Path to the current active experiment's directory.
        _config_snapshot (dict): A dictionary holding the configuration snapshot for the current experiment.
    """

    def __init__(self, experiments_root_dir: pathlib.Path, experiment_name_prefix: str = "exp"):
        """
        Initializes the ExperimentManager.

        Args:
            experiments_root_dir: The root directory for all experiments.
            experiment_name_prefix: Prefix for naming created experiment directories.
        """
        self.experiments_root_dir = experiments_root_dir.resolve() # Ensure it's an absolute path
        self.experiment_name_prefix = experiment_name_prefix
        self._current_experiment_path: pathlib.Path | None = None
        self._config_snapshot: dict = {}
        self.logger: logging.Logger | None = None

    def _setup_experiment_logger(self) -> None:
        """
        Sets up a dedicated logger for the current experiment.
        
        The logger will output to a file named 'experiment_run.log' within the 
        current experiment directory. If no experiment path is set, it falls back
        to a basic console logger with a warning.
        This method ensures that duplicate handlers are not added if called multiple times.
        """
        if not self._current_experiment_path: # Use internal attribute for check
            # Fallback logger setup
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(f"exp_manager_fallback_{datetime.datetime.now().timestamp()}") # Unique fallback logger name
            self.logger.warning("Logger setup: No experiment path set. Using fallback basic console logger.")
            return

        log_file_name = "experiment_run.log"
        log_file_path = self.current_experiment_path / log_file_name

        logger_name = f"exp_{self.current_experiment_path.name}" # Unique logger per experiment
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # Avoid duplicate logs from root logger if it's configured

        # Remove existing handlers to prevent duplication if re-initializing logger for the same path
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        # File Handler for experiment-specific log file
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(f"Logger initialized. Logging to: {log_file_path}")

    def create_new_experiment(self, timestamp_format: str = "%Y-%m-%d_%H-%M-%S") -> pathlib.Path:
        """
        Creates a new timestamped directory for an experiment run.
        Initializes the experiment-specific logger after directory creation.

        Args:
            timestamp_format: The format string for the timestamp in the directory name.

        Returns:
            pathlib.Path: The path to the newly created experiment directory.
        """
        timestamp = datetime.datetime.now().strftime(timestamp_format)
        dir_name = f"{timestamp}_{self.experiment_name_prefix}"
        self._current_experiment_path = self.experiments_root_dir / dir_name
        self._current_experiment_path.mkdir(parents=True, exist_ok=True)
        
        self._setup_experiment_logger() # Setup logger immediately after path is confirmed
        
        # Use self.logger which is now guaranteed to be set (either specific or fallback)
        self.logger.info(f"Created new experiment directory: {self._current_experiment_path}")
        return self._current_experiment_path

    @property
    def current_experiment_path(self) -> pathlib.Path:
        """Provides the path to the currently active experiment directory."""
        if not self._current_experiment_path:
            # This situation should ideally be prevented by workflow logic.
            # Logging here before raising can be helpful for debugging if it occurs.
            if self.logger: 
                self.logger.error("Attempted to access current_experiment_path before it was set.")
            else: # Fallback if logger itself isn't even set
                logging.error("Attempted to access current_experiment_path before it (and its logger) was set.")
            raise ValueError("No experiment has been created or set. Call create_new_experiment() or set_current_experiment_path().")
        return self._current_experiment_path
    
    def get_data_split_dir(self, data_split_dir_name: str) -> pathlib.Path:
        """
        Gets the path to the data split directory within the current experiment.
        Creates the directory if it doesn't exist.

        Args:
            data_split_dir_name: The name of the subdirectory for data splits.

        Returns:
            pathlib.Path: Path to the data split directory.
        """
        path = self.current_experiment_path / data_split_dir_name
        path.mkdir(parents=True, exist_ok=True)
        if self.logger: self.logger.debug(f"Data split directory ensured at: {path}")
        return path

    def get_finetuned_model_dir(self, finetuned_model_dir_name: str) -> pathlib.Path:
        """
        Gets the path to the finetuned model directory within the current experiment.
        Creates the directory if it doesn't exist.

        Args:
            finetuned_model_dir_name: Name of the subdirectory for finetuned models.

        Returns:
            pathlib.Path: Path to the finetuned model directory.
        """
        path = self.current_experiment_path / finetuned_model_dir_name
        path.mkdir(parents=True, exist_ok=True)
        if self.logger: self.logger.debug(f"Finetuned model directory ensured at: {path}")
        return path

    def get_lora_checkpoint_path(self, finetuned_model_dir_name: str, lora_checkpoint_name: str) -> pathlib.Path:
        """
        Constructs the path to a specific LoRA checkpoint file or directory.

        Args:
            finetuned_model_dir_name: The name of the parent finetuned model directory.
            lora_checkpoint_name: The name of the LoRA checkpoint file/subdir.

        Returns:
            pathlib.Path: Full path to the LoRA checkpoint.
        """
        return self.get_finetuned_model_dir(finetuned_model_dir_name) / lora_checkpoint_name

    def get_evaluation_results_dir(self) -> pathlib.Path:
        """
        Gets the path to the evaluation results directory within the current experiment.
        Creates the directory if it doesn't exist.

        Returns:
            pathlib.Path: Path to the evaluation results directory.
        """
        path = self.current_experiment_path / "evaluation_results"
        path.mkdir(parents=True, exist_ok=True)
        if self.logger: self.logger.debug(f"Evaluation results directory ensured at: {path}")
        return path

    def get_config_snapshot_path(self) -> pathlib.Path:
        """Gets the path where the experiment's configuration snapshot will be saved."""
        return self.current_experiment_path / "experiment_config.json"

    def save_config_snapshot(self, config_module) -> None:
        """
        Saves a snapshot of the given configuration module as a JSON file 
        in the current experiment directory.

        Args:
            config_module: The Python module whose attributes will be serialized to JSON.
        """
        if not self.logger: self._setup_experiment_logger() # Should be redundant if workflow is correct
        
        target_path = self.get_config_snapshot_path()
        config_dict = {}
        for key in dir(config_module):
            if not key.startswith('__') and not callable(getattr(config_module, key)):
                value = getattr(config_module, key)
                if isinstance(value, pathlib.Path):
                    config_dict[key] = str(value)
                elif isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                    # More robust handling for common serializable types
                    config_dict[key] = value
                else:
                    # For other types, convert to string and log a warning
                    config_dict[key] = str(value)
                    if self.logger: self.logger.warning(f"Config key '{key}' with value '{value}' of type {type(value)} was converted to string for JSON serialization.")
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            self.logger.info(f"Configuration snapshot saved to {target_path}")
            self._config_snapshot = config_dict
        except Exception as e:
            self.logger.error(f"Error saving configuration snapshot to {target_path}: {e}", exc_info=True)

    def load_experiment_config(self, experiment_path: pathlib.Path | None = None) -> dict:
        """
        Loads the 'experiment_config.json' from a specified experiment path or the current one.
        Updates `_config_snapshot` with the loaded configuration.

        Args:
            experiment_path: Optional path to an experiment directory. If None, uses the current experiment path.

        Returns:
            A dictionary with the loaded configuration, or an empty dict if not found or error occurs.
        """
        if not self.logger: self._setup_experiment_logger()

        path_to_load_from = experiment_path or self._current_experiment_path # Use internal attribute
        if not path_to_load_from:
             self.logger.error("load_experiment_config: Path not specified and no current experiment set.")
             raise ValueError("Experiment path not specified and no current experiment set.")

        config_file = path_to_load_from / "experiment_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._config_snapshot = json.load(f)
                self.logger.info(f"Loaded experiment configuration from {config_file}")
                return self._config_snapshot
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {e}", exc_info=True)
                self._config_snapshot = {} # Reset on error
                return {}
        else:
            self.logger.warning(f"Experiment config snapshot not found at {config_file}. No config loaded.")
            self._config_snapshot = {} # Reset if not found
            return {}
        
    def get_config_value(self, key: str, default_value: any = None, main_config_module = None) -> any:
        """
        Retrieves a configuration value.
        
        Priority order:
        1. From the loaded experiment-specific config snapshot (`_config_snapshot`).
        2. From the provided `main_config_module` (e.g., global `src.config`).
        3. The `default_value`.

        Args:
            key: The configuration key to retrieve.
            default_value: The value to return if the key is not found in any config.
            main_config_module: The global/fallback Python config module.

        Returns:
            The retrieved configuration value.
        """
        value = self._config_snapshot.get(key)
        if value is not None:
            return value
        if main_config_module and hasattr(main_config_module, key):
            return getattr(main_config_module, key)
        return default_value

    def set_current_experiment_path(self, path: pathlib.Path) -> None:
        """
        Sets the current experiment directory path and initializes its logger.
        Also attempts to load the configuration snapshot from this path.
        Typically used when an evaluation is run on a pre-existing experiment directory.

        Args:
            path: The path to an existing experiment directory.

        Raises:
            FileNotFoundError: If the provided path is not a directory.
        """
        resolved_path = path.resolve()
        if not resolved_path.is_dir():
            # Log this critical error before raising, as logger setup depends on a valid path.
            # A general logger might be the only option if self.logger is not yet configured.
            logging.error(f"Attempted to set experiment path to a non-existent directory: {resolved_path}")
            raise FileNotFoundError(f"Experiment directory not found: {resolved_path}")
            
        self._current_experiment_path = resolved_path
        self._setup_experiment_logger() # Setup logger based on the new path
        
        self.logger.info(f"Set current experiment path to: {self._current_experiment_path}")
        self.load_experiment_config() # Load config for this (now current) experiment 
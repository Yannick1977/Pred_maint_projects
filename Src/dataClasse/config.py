from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class config_path:
    work_dir: Path
    model_dir: Path
    local_data_file: Path
    local_data_file_analyse: Path
    local_data_file_validation: Path
    local_data_file_train: Path
    local_data_file_test: Path
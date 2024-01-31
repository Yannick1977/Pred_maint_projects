import os
from Src.dataClasse.config import config_path
from Src.utils.common import read_yaml,create_directories
from pathlib import Path


class config_manager:
    def __init__(self):
        #if (os.getcwd().split('\\')[-1])=='Notebook':
        #    os.chdir('../')

        self.cfg = read_yaml(Path("Config\path.yaml"))
        create_directories([self.cfg.config_path.work_dir])
        create_directories([self.cfg.config_path.model_dir])
    
    def get_path(self)->config_path:
        return self.cfg
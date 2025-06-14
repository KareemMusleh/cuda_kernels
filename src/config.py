import yaml
import os
from pathlib import Path

yaml_config = yaml.safe_load(open(Path(__file__).parent.parent/'config.yaml'))

# Load the env variables
os.environ.update(yaml_config.pop('env'))

# Load the local variables
vars().update(yaml_config)
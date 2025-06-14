import yaml
import os
from pathlib import Path

yaml_config = yaml.safe_load(open(Path(__file__).parent.parent/'config.yaml'))


# Some of the variables in the config should be environment variables
os.environ.update(yaml_config.pop('env'))

# The rest are local variables
vars().update(yaml_config)
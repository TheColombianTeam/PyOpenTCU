import os
from datetime import datetime
from omegaconf import OmegaConf

# Retrieve the configs path
conf_path = os.path.join(os.path.dirname(__file__), "../configs")

# Retrieve the default config
args = OmegaConf.load(os.path.join(conf_path, "default.yaml"))

# Read the cli args
cli_args = OmegaConf.from_cli()

# read a specific config file
if "config" in cli_args and cli_args.config:
    conf_args = OmegaConf.load(cli_args.config)
    args = OmegaConf.merge(args, conf_args)
# else:
# raise NotImplementedError('You should introduce the config')

# Merge cli args into config ones
args = OmegaConf.merge(args, cli_args)

# add log directories
args.log_dir = os.path.join(args.name)
args.logfile = os.path.join(
    args.log_dir, datetime.now().strftime("%b%d_%H-%M-%S") + ".log"
)

os.makedirs(args.log_dir, exist_ok=True)

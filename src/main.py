import argparse
import yaml

from .app.whatif import WhatIf


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to a config file")
args = vars(parser.parse_args())

with open(args["config_path"], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

WhatIf(config)

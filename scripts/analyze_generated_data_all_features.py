import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to a config file")
parser.add_argument("mode", type=str, choices=["all", "clip", "remove"], default="all",
                    help="How time series with observations outside the min and max values are filtered."
                         "'all' to apply no filter,"
                         "'clip' to clip observations between the min and max seen in test data,"
                         "'remove' to remove time series with observations higher than seen in test data.")

args = vars(parser.parse_args())
for feature in ["trend_str", "trend_slope", "trend_lin", "seasonal_str"]:
    subprocess.run(["python", "scripts/analyze_generated_data.py", args["config_path"], feature, args["mode"]])

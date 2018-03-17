import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath", type=str, help="Filepath to checkpoint.")  # ...checkpoints/model-XYZ
args = parser.parse_args()

print("Loading checkpoint...")
reader = tf.train.NewCheckpointReader(args.filepath)
print("Loaded checkpoint.")

print("Counting the number of parameters...")
param_map = reader.get_variable_to_shape_map()
total_count = 0
for k, v in param_map.items():
    if "Adam" in k or "beta" in k or "global_step" in k:
        pass
    else:
        temp = np.prod(v)
        total_count += temp
        print("{}: {} => {}".format(k, str(v), temp))

print("Total Param Count: {}".format(total_count))

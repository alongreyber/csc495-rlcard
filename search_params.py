import itertools
import subprocess

# Automated grid search experiments
mlp_layer_counts = [1,2,3,4]
mlp_layer_sizes = [8, 16, 32, 64, 128, 256]

# Iterate over all combinations of hyperparameter values.
for mlp_layer_count, mlp_layer_size in itertools.product(mlp_layer_counts, mlp_layer_sizes):
    # Execute "dvc exp run --queue --set-param train.n_est=<n_est> --set-param train.min_split=<min_split>".
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"src/params.py:TrainConfig.mlp_layer_count={mlp_layer_count}",
                    "--set-param", f"src/params.py:TrainConfig.mlp_layer_size={mlp_layer_size}"])

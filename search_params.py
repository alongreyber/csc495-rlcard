import itertools
import subprocess

# Automated grid search experiments
mlp_layer_counts = [2,4]
mlp_layer_sizes = [32, 64, 128, 256]
learning_rates = [5e-6, 5e-5, 5e-4]

# Iterate over all combinations of hyperparameter values.
for mlp_layer_count, mlp_layer_size, learning_rate in itertools.product(mlp_layer_counts, mlp_layer_sizes, learning_rates):
    # Execute "dvc exp run --queue --set-param train.n_est=<n_est> --set-param train.min_split=<min_split>".
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"src/params.py:TrainConfig.mlp_layer_count={mlp_layer_count}",
                    "--set-param", f"src/params.py:TrainConfig.mlp_layer_size={mlp_layer_size}"
                    "--set-param", f"src/params.py:TrainConfig.learning_rate={learning_rate}",
                    ])

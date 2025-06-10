"""
Config file: parameters can be adjusted in this file
"""

seed = 42
batch_size = 64
n_epochs = 200
latent_dimension = 2
input_dim = 60
num_samples = 5000
learning_rate = 1e-3
validation_split = 0.2

# Path to the files
input_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/input_data.csv"
latent_output_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/latent_variables.csv"
generated_output_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations.csv"
generated_output_denorm_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations_denorm.csv"
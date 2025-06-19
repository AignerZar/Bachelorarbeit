"""
Config file: parameters can be adjusted in this file
"""
import torch

seed = 42
batch_size = 16
n_epochs = 1000
latent_dimension = 150
latent_dimension_test = 2 #for testing a different set of hyperparameters
input_dim = 60
num_samples = 4949
learning_rate = 1e-3
validation_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the files -> replace for own use, the test files were used for testing different model architectures
input_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/input_data.csv"
latent_output_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/latent_variables.csv"
generated_output_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations.csv"
generated_output_denorm_file = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations_denorm.csv"
generated_output_denorm_file_test = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations_denorm_test.csv"
generated_output_file_test = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/generated_configurations_test.csv"
latent_output_file_test = "/Users/zarahaigner/Documents/Physik_6 Semester/Bachelorthesis/Git-Repo/latent_variables_test.csv"

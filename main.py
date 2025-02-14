import os
import sys
import utils.utils as ut
import utils.eda as eda
import model.RunModel as RunModel
import loader.loader as loader
import gen_dataset.GenDataset as GenDataset
from utils.ConfigLogger import ConfigLogger


def main():
    """
    Main function to execute the pipeline including configuration logging, data loading,
    dataset generation, exploratory data analysis, and preparing for model training.

    The function performs the following steps:
    - Logs the configuration to a CSV file.
    - Loads raw data using the data loader.
    - Generates a processed dataset using the GenDataset module.
    - Saves the generated dataset to a CSV file.
    - Executes exploratory data analysis (EDA) on the dataset.
    - (Commented out) Prepares the pipeline for model training and evaluation.

    Raises:
        IOError: If any file operation fails, such as reading, writing, or saving the dataset
        and configurations.
        ValueError: If any of the intermediate process steps encounter invalid input or format.
    """
    # Saving configuration execution in 'data/config_history.csv'
    config_logger = ConfigLogger(output_dir='data')
    config_logger.log_config()
    config = ut.load_config('main_config')

    dataframes = loader.run_loader()
    df_aqf = GenDataset.run_gen_dataset(dataframes)

    df_aqf.to_csv('data/df_aqf.csv',encoding='utf-8',index=False)

    if config.get('exec_eda', False):
        eda.run_eda(df_aqf)

    x_train, x_test, y_train, y_test = RunModel.run_model(df_aqf)

    x_train.to_csv('data/x_train.csv',encoding='utf-8',index=False)
    x_test.to_csv('data/x_test.csv',encoding='utf-8',index=False)


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
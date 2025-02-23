import os
import sys
import utils.utils as ut
import utils.eda as eda
import model.model as model
import loader.loader as loader
import gen_dataset.gen_dataset as gen_dataset

def main():
    """
    Executes the main program workflow, which includes configuration logging, data
    loading, dataset generation, optional exploratory data analysis, model execution,
    and saving the generated dataset.

    This function facilitates end-to-end processing by managing the sequence of key
    steps such as initializing the configuration logger, running data loaders,
    generating datasets, optionally performing exploratory data analysis if
    configured, and executing the desired model logic. The generated dataset
    is saved to a CSV file for further usage.

    Parameters:
        No parameters are required for this function.

    Returns:
        None
    """
    config = ut.load_config('main_config')

    dataframes = loader.run_loader()
    df_aqf = gen_dataset.run_gen_dataset(dataframes)

    df_aqf.to_csv('data/df_aqf.csv',encoding='utf-8',index=False)

    if config.get('exec_eda', False):
        eda.run_eda(df_aqf)

    model.run_model(df_aqf)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
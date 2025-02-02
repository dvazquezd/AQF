import os
import sys
import loader.loader as loader
import gen_dataset.GenDataset as GenDataset
from utils.ConfigLogger import ConfigLogger
import utils.eda as eda

def main():
    """
    """
    # Guardar la configuración de esta ejecución en 'data/config_history.csv'
    config_logger = ConfigLogger(output_dir='data')
    config_logger.log_config()

    dataframes = loader.run_loader()
    df_aqf = GenDataset.run_gen_dataset(dataframes)

    df_aqf.to_csv('data/df_aqf.csv',encoding='utf-8',index=False)

    #eda.run_eda(df_aqf)


    # 3. Entrenar y evaluar modelo
    #model = model_train.train_and_evaluate(dataset)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
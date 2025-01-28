import os
import sys
import loader.loader as loader
import gen_dataset.GenDataset as GenDataset
#import models.scripts.train as model_train

def main():
    """
    """
    dataframes = loader.run_loader()
    tec_info, news_info = GenDataset.run_gen_dataset(dataframes)

    tec_info.to_csv('tec_info.csv',index=False)
    news_info.to_csv('news_info.csv',index=False)

    # 2. Generar dataset final
    #dataset = dataset_gen.generate_dataset(dataframes)

    # 3. Entrenar y evaluar modelo
    #model = model_train.train_and_evaluate(dataset)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
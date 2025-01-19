import os
import sys
import loader.loader as loader
import gen_dataset.GenDataset as gen
#import dataset_generator.generator as dataset_gen
#import models.scripts.train as model_train

def main():
    """
        main()

        This function orchestrates the overall workflow by loading data, generating dataset
        from the processed data, and exporting the final dataset to a CSV file. Each step is
        performed sequentially to ensure proper data handling and processing for further use.

        Returns:
            None: This function does not return any data.
    """
    dataframes = loader.run_loader()

    tec_info = gen.run_gen_dataset(dataframes)

    tec_info.to_csv('tec_info.csv',index=False)



    # 2. Generar dataset final
    #dataset = dataset_gen.generate_dataset(dataframes)

    # 3. Entrenar y evaluar modelo
    #model = model_train.train_and_evaluate(dataset)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
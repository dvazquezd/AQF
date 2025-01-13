import os
import sys
import loader.loader as loader
import gen_dataset.gen_dataset as gen
#import dataset_generator.generator as dataset_gen
#import models.scripts.train as model_train

def main():
    # 1. Cargar y procesar datos iniciales
    dataframes = loader.run_loader()

    for key in dataframes.keys():
        print(f'Dataframe: {key}\n')
        print(dataframes[key].describe())
        print(dataframes[key].info())
        print(dataframes[key].head())



    #config = gen.run_gen_dataset()

    #print(config['features'])

    # 2. Generar dataset final
    #dataset = dataset_gen.generate_dataset(dataframes)

    # 3. Entrenar y evaluar modelo
    #model = model_train.train_and_evaluate(dataset)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import get_time_now

def inspect_dataset(df):
    """
    Muestra información básica sobre el dataset.
    """
    print(f'\n{get_time_now()} :: EDA: Dataset general info:')
    print(df.info())

    print("\n Estadísticas descriptivas:")
    print(df.describe())

    print("\n Valores nulos por columna:")
    print(df.isnull().sum())

def plot_target_distribution(df):
    """
    Grafica la distribución del target (0 = baja, 1 = sube).
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["target"], hue=df["target"], palette="coolwarm", legend=False)
    plt.title("Distribución del Target (0 = Baja, 1 = Sube)")
    plt.xlabel("Target")
    plt.ylabel("Cantidad de Registros")
    plt.show()

def plot_correlation_matrix(df):
    """
    Grafica la matriz de correlación entre variables numéricas.
    """
    # Filtrar solo columnas numéricas
    numeric_df = df.select_dtypes(include=['number'])

    # Verificar que haya columnas numéricas antes de graficar
    if numeric_df.empty:
        print("No hay columnas numéricas en el dataset para calcular la correlación.")
        return

    # Calcular y graficar la matriz de correlación
    plt.figure(figsize=(12, 6))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, annot_kws={"size": 8})

    plt.title("Matriz de Correlación de Variables Numéricas")
    plt.show()

def plot_price_trend(df):
    """
    Grafica la evolución del precio de cierre.
    """
    plt.figure(figsize=(12, 5))
    df["datetime"] = pd.to_datetime(df["datetime"])
    plt.plot(df["datetime"], df["close"], label="Precio de Cierre", color="blue")
    plt.title("Evolución del Precio de Cierre")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_technical_indicators(df):
    """
    Grafica histogramas de indicadores técnicos.
    """
    indicators = ["rsi_5", "rsi_7", "rsi_9", "sma_5", "sma_10", "sma_12", "MACD"]
    if indicators[0] in df.columns:
        df[indicators].hist(figsize=(12, 8), bins=30, edgecolor="black")
        plt.suptitle("Distribución de Indicadores Técnicos", fontsize=14)
        plt.show()

def plot_economic_indicators(df):
    """
    """
    economic_vars = ["cpi", "unemployment", "nonfarm_payroll"]
    if all(col in df.columns for col in economic_vars):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[economic_vars])
        plt.title("Boxplot de Indicadores Económicos")
        plt.xticks(rotation=20)
        plt.show()

def plot_sentiment_vs_target(df):
    """
    """
    if "ticker_ssm" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=df["ticker_ssm"], y=df["target"], alpha=0.5)
        plt.title("Relación entre Sentimiento y Target (Sube/Baja)")
        plt.xlabel("Sentimiento Promedio del Ticker")
        plt.ylabel("Target (0 = Baja, 1 = Sube)")
        plt.show()
    else:
        print("❌ La columna 'ticker_ssm' no está en el dataset.")

def run_eda(df):
    """
    Ejecuta el análisis exploratorio de datos (EDA).
    """
    if df is not None:
        inspect_dataset(df)
        #plot_target_distribution(df)
        plot_correlation_matrix(df)
        plot_price_trend(df)
        plot_technical_indicators(df)
        plot_economic_indicators(df)
        plot_sentiment_vs_target(df)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    """
    Chargement, encodage et mise à l'échelle du dataset INCA2.
    """
    def __init__(self, csv_path, sep=';'):
        self.csv_path = csv_path
        self.sep = sep
        self.encoders = {}
        self.scaler = StandardScaler()

    def load(self):
        df = pd.read_csv(self.csv_path, sep=self.sep, encoding='latin-1')
        return df

    def fit_transform(self, df, numeric_cols, categorical_cols):
        # Encodage des catégorielles
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        # Normalisation des numériques
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def transform(self, df):
        for col, le in self.encoders.items():
            df[col] = le.transform(df[col].astype(str))
        df[df.select_dtypes(include=[np.number]).columns] = \
            self.scaler.transform(df.select_dtypes(include=[np.number]))
        return df

from .data_preprocessing import DataPreprocessor
from .tabular_gan       import TabularGANAugmentor
from .rdf2vec_embedder  import RDF2VecEmbedder
from .shacl_validator   import SHACLValidator
import pandas as pd

class Pipeline:
    """
    Pipeline complet :
      1) Charger et prétraiter INCA2
      2) Entraîner le GAN custom et générer des échantillons synthétiques
      3) Extraire des embeddings RDF2Vec
      4) Valider le KG via SHACL avec inference='owlrl'
    """

    def __init__(self, csv_path: str, kg_ttl_path: str, shapes_ttl_path: str):
        self.csv_path      = csv_path
        self.kg_ttl_path   = kg_ttl_path
        self.shapes_ttl    = shapes_ttl_path
        self.prep      = DataPreprocessor(self.csv_path)
        self.gan       = TabularGANAugmentor()
        self.rdf2vec   = RDF2VecEmbedder(self.kg_ttl_path)
        self.validator = SHACLValidator(self.shapes_ttl)

    def run(self,
            numeric_cols: list[str],
            categorical_cols: list[str],
            num_augmented_samples: int
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
        #Chargement & Prétraitement
        df_raw  = self.prep.load()
        df_proc = self.prep.fit_transform(df_raw.copy(),
                                          numeric_cols,
                                          categorical_cols)

        #Entraînement + Génération GAN
        self.gan.fit(df_proc,
                     categorical_columns=categorical_cols)
        df_synth = self.gan.sample(num_augmented_samples)

        # Embeddings RDF2Vec 
        self.rdf2vec.fit()
     

        #Validation SHACL avec owlrl
        conforms, report = self.validator.validate(
            data_ttl=self.kg_ttl_path,
            inference='owlrl'  
        )
        if conforms:
            print("SHACL validation passed")
        else:
            print("SHACL validation errors:\n", report)

        return df_proc, df_synth

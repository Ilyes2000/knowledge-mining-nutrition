# nutrition_recommender/shacl_validator.py

from pyshacl import validate

class SHACLValidator:
    """
    Validation SHACL pour garantir la cohérence métier.
    """

    def __init__(self, shapes_path: str):
        self.shapes = shapes_path

    def validate(self,
                 data_ttl: str,
                 inference: str = 'owlrl' 
                ) -> tuple[bool, str]:
        """
        data_ttl  : chemin vers le fichier TTL à valider
        inference : 'none', 'rdfs' ou 'owlrl'
        """
        conforms, results_graph, results_text = validate(
            data_graph=data_ttl,
            shacl_graph=self.shapes,
            inference=inference,
            serialize_report_graph=True
        )
        return conforms, results_text

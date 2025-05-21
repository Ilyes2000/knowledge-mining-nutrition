<p align="center">
  <!-- Python -->
  <img alt="Python" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" height="55">&nbsp;&nbsp;
  <!-- PyTorch -->
  <img alt="PyTorch" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg" height="55">&nbsp;&nbsp;

  <!-- Jupyter -->
  <img alt="Jupyter" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg" height="55">&nbsp;&nbsp;
  <!-- Pandas (NOUVEAU lien) -->
  <img alt="Pandas" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" height="55">&nbsp;&nbsp;
  
  <!-- OpenAI : utiliser le logo minimal en SVG -->
  <img alt="OpenAI" src="https://avatars.githubusercontent.com/u/63457326?s=200&v=4" height="55">&nbsp;&nbsp;
  <!-- Git -->
  <img alt="Git" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/git/git-original.svg" height="55">&nbsp;&nbsp;
  <!-- GitHub -->
  <img alt="GitHub" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/github/github-original.svg" height="55">
</p>

<br/>


# knowledge-mining-nutrition
> Data augmentation & personalised food-recommendation for the INCA 2 survey  
> **Université Paris-Saclay – LISN – TER 2025**

---

## 1. Contexte

Le jeu de données **INCA 2** (_~540 000 prises alimentaires – 4 079 individus – 7 jours_) présente :

* une **sparsité** marquée sur certains champs (`codal`, `nojour`) ;
* aucun _profil_ nutritionnel directement exploitable.

Notre objectif : **enrichir** ces données via un GAN tabulaire + graphe de connaissances, puis **classifier** chaque repas (végétarien, diabétique, obèse, sain) et générer une recommandation textuelle grâce à un LLM.

---

## 2. Arborescence du dépôt

knowledge-mining-nutrition/
├── data_augmented/               # CSV synthétiques générés
│   └── …                         
├── notebooks/
│   ├── train.ipynb               # exécution complète (00→03)
│   └── checkpoints/              # poids .pth sauvegardés
│       └── *.pth
├── nutrition_recommender/        # package Python principal
│   ├── __init__.py
│   ├── cvae.py                   # (option) auto-encodeur conditionnel
│   ├── data_preprocessing.py     # encodage / scaling INCA2
│   ├── pipeline.py               # orchestration end-to-end
│   ├── rdf2vec_embedder.py       # embeddings KG avec rdf2vec
│   ├── shacl_validator.py        # validation structurelle SHACL
│   └── tabular_gan.py            # implémentation GAN tabulaire
├── resources/
│   ├── Dataset_INCA2/            # dump ANSES (non versionné – .gitignore)
│   └── kg/                       # ontologie, contraintes, shapes
├── LICENSE
└── README.md


## 3. Installation rapide

git clone https://github.com/Ilyes2000/knowledge-mining-nutrition.git
cd knowledge-mining-nutrition
python -m venv .venv && source .venv/bin/activate   # (Windows : venv\Scripts\Activate)
pip install -r requirements.txt

# Clé OpenAI uniquement en local :
echo "OPENAI_API_KEY=sk-..." > .env



## 4. Citation

@misc{sais2025inca2gan,
  author = {Ilyes Sais},
  title  = {Knowledge mining & nutrition: data augmentation with Tabular GAN},
  year   = {2025},
  url    = {https://github.com/Ilyes2000/knowledge-mining-nutrition}
}





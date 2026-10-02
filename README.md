# Classification de Commentaires Citoyens — Afriklang x Togo AI Lab

Ce projet est réalisé par **NAMESSI Mazama-Esso** dans le cadre du test technique de sélection pour le stage chez **Afriklang** en partenariat avec **Togo AI Lab** (TAISS 2026).

## 📌 Présentation de l'Approche
L'objectif est d'automatiser la classification de commentaires citoyens sur les services publics en 3 catégories : **Satisfaction**, **Insatisfaction**, et **Suggestion**.

1. **Exploration & Prétraitement** :
   - Analyse de la distribution des classes (dataset équilibré de 150 commentaires) et de la longueur des textes.
   - Nettoyage du texte (minuscules, retrait des chiffres/ponctuation).
   - Préservation des mots clés de négation/polarité et des termes en langues locales Éwé/Mina (*akpe*, *nyuie*, *mègbe*).
2. **Vectorisation & Modélisation** :
   - Vectorisation **TF-IDF** avec prise en compte des uni-grammes et bi-grammes.
   - Évaluation comparative de 4 algorithmes : *Random Forest*, *SVM Linéaire*, *Régression Logistique*, et *Naive Bayes*.
   - Découpage stratifié 80% Entraînement / 20% Test (`random_state=42`).

## 📊 Synthèse des Résultats

| Modèle | Accuracy | F1-Score Macro |
| :--- | :---: | :---: |
| **Random Forest** | **76.67%** | **0.7660** |
| **SVM (Linéaire)** | 73.33% | 0.7368 |
| **Régression Logistique** | 73.33% | 0.7363 |
| **Naive Bayes** | 70.00% | 0.7047 |

## 🛠️ Instructions d'Installation et d'Exécution

```bash
# 1. Cloner le dépôt GitHub
git clone [https://github.com/](https://github.com/)[votre-username]/Test-NLP-Afriklang.git
cd Test-NLP-Afriklang

# 2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le notebook
jupyter notebook notebook.ipynb

# 5. (Bonus) Lancer l'interface démo Streamlit
streamlit run app.py

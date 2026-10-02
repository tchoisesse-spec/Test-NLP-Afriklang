import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Afriklang — Classification Commentaires", page_icon="🇹🇬", layout="centered")

st.title("🇹🇬 Afriklang x Togo AI Lab")
st.subheader("Classification automatique de retours citoyens")

@st.cache_resource
def load_and_train():
    df = pd.read_csv('dataset_nlp_test_tal.csv')
    
    FRENCH_STOPWORDS = set([
        'a', 'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux',
        'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi',
        'mon', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pour', 'qu', 'que', 'qui', 'sa', 'se',
        'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous',
        'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'est', 'sommes', 'êtes', 'sont'
    ])
    PRESERVED_WORDS = {'pas', 'non', 'ne', 'plus', 'jamais', 'rien', 'sans', 'trop', 'bien', 'très', 'akpe', 'nyuie', 'mele', 'mègbe'}
    CUSTOM_STOPWORDS = FRENCH_STOPWORDS - PRESERVED_WORDS
    
    def clean(text):
        text = text.lower()
        text = re.sub(r'[^a-zàâäéèêëîïôöùûüçɔɛŋ\s]', ' ', text)
        tokens = [t for t in text.split() if t not in CUSTOM_STOPWORDS and len(t) > 1]
        return " ".join(tokens)
    
    df['clean'] = df['texte'].apply(clean)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(df['clean'])
    y = df['categorie']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    return vec, clf, clean

vec, clf, clean_func = load_and_train()

user_input = st.text_area("Entrez un commentaire citoyen :", "Le service en ligne est très fluide et rapide, akpe kaka !")

if st.button("Analyser la catégorie"):
    if user_input.strip() != "":
        cleaned = clean_func(user_input)
        vec_input = vec.transform([cleaned])
        pred = clf.predict(vec_input)[0]
        probs = clf.predict_proba(vec_input)[0]
        
        st.markdown(f"### Catégorie Prédite : **{pred}**")
        
        prob_df = pd.DataFrame({
            'Catégorie': clf.classes_,
            'Probabilité': [f"{p*100:.1f}%" for p in probs]
        })
        st.table(prob_df)
    else:
        st.warning("Veuillez saisir un texte à analyser.")
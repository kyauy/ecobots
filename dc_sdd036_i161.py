import streamlit as st
from streamlit.components.v1 import html
from streamlit_chat import message
import pandas as pd
from PIL import Image
import nltk
from src.intents import INTENTS, image_dict, sound_dict
from src.utils import (
    query,
    submit,
    model_training,
    get_one_time_list,
    get_data,
    personalize_intents,
    my_html,
)

# Configuration page avec consignes

st.set_page_config(
    page_title="ECOS SDD 36 - M. Ledoux",
    page_icon=":robot:",
    initial_sidebar_state="expanded",
)

nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

image_pg = Image.open("img/ecobots.png")
st.sidebar.image(image_pg, caption=None, width=100)
st.sidebar.header("ECOS Chatbot: Patient simulé par l'intelligence artificielle")
st.sidebar.markdown(
    """
- *Contexte:*

Vous êtes interne aux urgences.
Le patient que vous allez prendre en charge se présente spontanément aux urgences.

M. Ledoux Jules, 65 ans présente des douleurs lombaires à droite depuis 4 jours. 
Il a consulté son médecin traitant il y a 2 jours à ce sujet qui lui a diagnostiqué une colique néphrétique. 
Il lui a prescrit du Ketoprofène et du Paracetamol. Les douleurs persistent malgré le traitement ce qui a poussé M. Ledoux à venir aux urgences.  
Nous sommes vendredi soir il est 21h.   
A l'arrivée ses constantes sont : TA 11/8, FC 107 bpm, température 38,8°C.
Vous n'avez pas à réaliser d'examen clinique. 

- *Objectifs:*

Vous devez évoquer une hypothèse diagnostique et prévoir les examens nécessaires pour la confirmer. 
Vous devez initier la prise en charge thérapeutique. 

Contact: [d-coudray@chu-montpellier.fr](mailto:d-coudray@chu-montpellier.fr) & [kevin.yauy@chu-montpellier.fr](mailto:kevin.yauy@chu-montpellier.fr)

"""
)

image_univ = Image.open("img/logosfacmontpellier.png")
st.sidebar.image(image_univ, caption=None, width=190)
image_chu = Image.open("img/CHU-montpellier.png")
st.sidebar.image(image_chu, caption=None, width=95)

# Functions


@st.cache(allow_output_mutation=True)
def get_intents():
    return INTENTS


@st.cache(allow_output_mutation=True)
def get_one_time_list_load(sdd):
    return get_one_time_list(sdd)


@st.cache(allow_output_mutation=True)
def personalize_intents_load(intents, patients_descriptions):
    return personalize_intents(intents, patients_descriptions)


@st.cache(allow_output_mutation=True)
def get_data_load(sdd, intents):
    return get_data(sdd, intents)


@st.cache(allow_output_mutation=True)
def model_training_load(data):
    return model_training(data)


def get_text():
    input_text = st.text_input(
        "Vous (interne aux urgences): ",
        key="input",
        help="Discutez avec votre patient avec des phrases complètes. Si problème: contactez kevin.yauy@chu-montpellier.fr",
        on_change=submit,
    )
    return st.session_state.answer


# JSON INPUT

## utilisation d'un dictionnaire pour représenter un fichier JSON d'intentions

image_dict_sdd = {
    "Attendez Docteur, on m'a déjà fait faire une prise de sang tout à l'heure. Peut etre que vous aurez déjà des resultats ? *** Voici les resultats que vous retrouvez sur le dossier informatisé du patient ***": "img/dc_sdd036_bio.png",
    "Mon médecin m'avait demandé de faire un scanner, j'ai eu de la chance j'ai pu avoir un rendez-vous ce matin, voilà le compte rendu.": "img/dc_sdd036_scan.png",
}

image_dict.update(image_dict_sdd)

patient_descriptions = {
    "name": ["M. Ledoux"],
    "taille": ["Je fais 1m75"],
    "poids": ["Je fais 92kg"],
    "job": ["Je suis retraité, ancien infirmier. "],
    "motif": [
        "J'ai mal au dos depuis 4 jours. J'ai vu mon médecin traitant il y a 4 jours il m'a dit que j'avais une colique néphrétique et m'a donné des antalgiques mais là ça ne va pas mieux ! "
    ],
    "stress": ["J'ai mal à mon dos.."],
    "age": ["J'ai 65 ans"],
    "antecedant": [
        "J'ai déjà eu une fois un crise de goutte pour lequel je prend de l'allopurinol."
    ],
    "antecedantfam": ["Pas à ma connaissance", "Je ne crois pas"],
    "howareyou": ["Vraiment pas bien Docteur."],
    "tabac": ["Non Docteur."],
}

sdd = [
    {
        "tag": "annonce",
        "patterns": [
            "Je comprends, d'après vos symptomes, vous avez probablement une pyélonéphrite aigue obstructive. Nous allons faire d'autres examens afin de pouvoir confirmer ou exclure cette hypothèse."
        ],
        "responses": [
            "Vous pouvez me detailler quels examens vous allez me prescrire docteur ? "
        ],
    },
    {
        "tag": "prescriptionExploration",
        "patterns": [
            "Je vais vous proposer de faire une prise de sang, avec une NFS, une CRP pour recherche une infection, une créatininémie pour evaluer votre fonction rénale. Vous devrez faire un examen urinaire et sanguin pour chercher des bactéries (ECBU et hémocultures). Enfin je vais vous proposer de réaliser un scanner abdominopelvien sans injection pour chercher la cause de vos douleurs et de cette probable pyelonephrite",
            "Je vais vous proposer une prise de sang afin de recherche si vous avez un syndrome inflammatoire. un ECBU mais aussi un scanner",
        ],
        "responses": [
            "Attendez Docteur, on m'a déjà fait faire une prise de sang tout à l'heure. Peut etre que vous aurez déjà des resultats ? *** Voici les resultats que vous retrouvez sur le dossier informatisé du patient ***"
        ],
    },
    {
        "tag": "analyseSangExploration",
        "patterns": [
            "Le bilan sanguin présente une insuffisance rénale légère et syndrome inflammatoire biologique "
        ],
        "responses": ["Qu'est ce qui faut faire encore ? "],
    },
    {
        "tag": "analyseImagerieExploration",
        "patterns": [
            "Vous allez faire un scanner abdominopelvien afin de voir la cause de vos fievres et vos douleurs.",
            "je vais vous prescrire un scanner abdominopelvien",
        ],
        "responses": [
            "Mon médecin m'avait demandé de faire un scanner, j'ai eu de la chance j'ai pu avoir un rendez-vous ce matin, voilà le compte rendu."
        ],
    },
    {
        "tag": "priseEnCharge",
        "patterns": [
            "Je vous confirme que vous souffrez d'une pyélonéphrite. Je vais vous hospitaliser et vous mettre rapidement sous antibiothérapie intraveineuse. Je vais contacter le médecin urologue, car il va falloir drainer vos voies urinaires, afin de guerir de votre pyelonéphrite. "
        ],
        "responses": ["Entendu Docteur, je vous fais confiance."],
    },
    {
        "tag": "repas",
        "patterns": ["Quel est l'heure de votre dernier repas ?"],
        "responses": ["Hier soir vers 20h Docteur."],
    },
]

# Generation des messages et du modèle d'IA

one_time_list = get_one_time_list_load(sdd)
intents = get_intents()
intents_perso = personalize_intents_load(intents, patient_descriptions)
data = get_data(sdd, intents_perso)
model, words, classes, lemmatizer = model_training_load(data)

st.header("Box 4 des urgences, 22h")

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "one_time_intent" not in st.session_state:
    st.session_state.one_time_intent = []

if "timer" not in st.session_state:
    st.session_state["timer"] = False

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False


placeholder = st.empty()
btn = placeholder.button(
    "Commencer l'ECOS", disabled=st.session_state.disabled, key="ECOS_go"
)

if btn:
    st.session_state["disabled"] = True
    st.session_state["timer"] = True
    placeholder.button(
        "Commencer l'ECOS", disabled=st.session_state.disabled, key="ECOS_running"
    )


if st.session_state["timer"] == True:
    html(my_html, height=50)

if st.button("Debug mode"):
    debug = "On"
else:
    debug = "Off"

user_input = get_text()

if user_input:
    output = query(
        user_input, debug, one_time_list, model, lemmatizer, words, classes, data
    )
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="pixel-art")
        if st.session_state["generated"][i] in image_dict.keys():
            st.image(image_dict[st.session_state["generated"][i]], caption=None)
        if st.session_state["generated"][i] in sound_dict.keys():
            st.audio(
                sound_dict[st.session_state["generated"][i]],
                format="audio/mp3",
                start_time=0,
            )
        message(
            st.session_state["past"][i],
            is_user=True,
            key=str(i) + "_user",
            avatar_style="pixel-art-neutral",
        )
    df = pd.DataFrame(
        list(zip(st.session_state["past"], st.session_state["generated"]))
    )
    df.columns = ["Vous", "Votre patient·e"]
    tsv = df.drop_duplicates().to_csv(sep="\t", index=False)
    st.download_button(
        label="Téléchargez votre conversation",
        data=tsv,
        file_name="conversation.tsv",
        mime="text/tsv",
    )

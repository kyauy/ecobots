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
    page_title="ECOS SDD 328 - M. Denis",
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

Vous êtes interne en neurologie.
Vous voyez en consultation Mr Denis, 55 ans, qui vient vous voir pour tremblement de repos du bras droit, ralentissement dans l'execution de ses gestes. 
L'examen clinique vous révèle un syndrome parkinsonien asymétrique typique d'une maladie de Parkinson. 
Vous posez le diagnostic de maladie de Parkinson. 
Il se présente seul à votre consultation.

- *Objectifs:*

Vous annoncez au patient le diagnostic de maladie de Parkinson, et les principaux signes sur lequel il repose
Vous en expliquez schématiquement la prise en charge thérapeutique.
A la fin de la station, vous lui proposez un traitement de 1re intention.

Contact: [m-villesot@chu-montpellier.fr](mailto:m-villesot@chu-montpellier.fr) & [kevin.yauy@chu-montpellier.fr](mailto:kevin.yauy@chu-montpellier.fr)

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
        "Vous (interne de neurologie): ",
        key="input",
        help="Discutez avec votre patient avec des phrases complètes. Si problème: contactez kevin.yauy@chu-montpellier.fr",
        on_change=submit,
    )
    return st.session_state.answer


# JSON INPUT

## utilisation d'un dictionnaire pour représenter un fichier JSON d'intentions

patient_descriptions = {'name': ['M. Denis'],
 'taille': ['Je fais 1m75'],
 'poids': ['Je fais 70kg'],
 'job': ["Je suis artiste peintre. J'aime mon métier."],
 'motif': ["Je viens vous consulter parce que j'ai peur d'avoir la maladie de Parkinson..."],
 'stress': ["J'ai peur d'avoir la maladie de Parkinson.."],
 'age': ["J'ai 55 ans"],
 'antecedant': ["J'ai de l'hypertension. Je prends de l'AMLOR le matin."],
 'antecedantfam': ['Pas à ma connaissance', 'Je ne crois pas'],
 'howareyou': ['Ca va Docteur.', 'Un peu stressée mais ca va.'],
 'tabac': ['Non Docteur.']}

sdd = [{'tag': 'annonce',
  'patterns': ['Vous avez bien fait, nous allons prendre le temps de vous expliquer la maladie de Parkinson.',
   'En effet, vous avez des symptomes pouvant faire penser à une maladie de Parkinson',
   "A l'examen, je retrouve des elements pouvant en effet indiquer que vous êtes atteint de la maladie de Parkison."],
  'responses': ['Je vous ecoute Docteur.',
   "Dites moi en plus Docteur s'il vous plait."]},
 {'tag': 'explorationParkinson',
  'patterns': ["Vous présentez des tremblements, une lenteur aux mouvements et une rigidité qui peut faire penser à une maladie de Parkinson. il s'agit d'une maladie qui est due à un déficit d'une molécule nommé dopamine dans le cerveau.",
   "A l'examen, je retrouve des elements cliniques, comme des tremblements et une rigidité de vos mouvements. Il s'agit d'une maladie qui est due à un déficit d'une molécule nommé dopamine dans le cerveau."],
  'responses': ["Mais vous êtes sur ? Y'a pas besoin d'examen complémentaires ?"]},
 {'tag': 'confirmationParkinson',
  'patterns': ["Tout à fait, le diagnostic est uniquement clinique. Il n'y a pas besoin de réaliser d'examen complémentaire. Le diagnostic de certitude est réalisé avec le suivi de l'absence d'autres signes pour un autre syndrome parkinsonien. Nous pourrons vous proposer un taritement dont la réponse nous orientera également sur le diagnostic.",
   "Tout à fait, le diagnostic est uniquement clinique. Il n'y a pas besoin de réaliser d'examen complémentaire. ",
   "Il n'y a pas besoin de réaliser d'examen complémentaire. Le diagnostic de certitude est réalisé avec le suivi de l'absence d'autres signes pour un autre syndrome parkinsonien. Nous pourrons vous proposer un taritement dont la réponse nous orientera également sur le diagnostic."],
  'responses': ["Est-ce que c'est grave ?"]},
 {'tag': 'pronosticParkinson',
  'patterns': ["Il s'agit d'une maladie dont l'evolution propre à chaque individu. Si on ne peut pas la guérir, on peut vous proposer des traitements pour essayer de réduire vos symptomes.",
   "Il s'agit d'une maladie dont l'evolution propre à chaque individu. Si on ne peut pas la guérir, on peut vous proposer des traitements pour essayer de réduire vos tremblements et vos lenteurs de mouvements."],
  'responses': ['Est-ce que je peux vivre normalement ?']},
 {'tag': 'traitementParkinson',
  'patterns': ["S'il n'existe pas de traitement pour guerir, il existe des traitements pour éviter vos symptomes, qui essaye de combler le déficit en dopamine qu'on retrouve dans la maladie de Parkinson. Ces traitementst sont des agonistes dopaminergiques comme la Levodopa. Ce traitement peut avoir des effets secondaires.",
   'Selon la gène causée par vos tremblements ou mouvements, je peux vous proposer des traitements pour soulager vos symptomes. Ces traitements ne permettent pas de guerir mais peuvent aider à calmer vos symptomes. Ils essayent de combler le déficit en dopamine, comme la levodopa qui est un agoniste dopaminergique. Ce traitement peut avoir des effets secondaires.'],
  'responses': ["Je prend un traitement pour la tension, est-ce qu'il y a un risque ?"]},
 {'tag': 'interactionParkinson',
  'patterns': ["En effet, il y a un risque d'hypotension de l'association traitement anti-parkinsonien et Amlor."],
  'responses': ['Je vais prendre le temps de réflechir avec vos explications Docteur.']}]

# Generation des messages et du modèle d'IA

one_time_list = get_one_time_list_load(sdd)
intents = get_intents()
intents_perso = personalize_intents_load(intents, patient_descriptions)
data = get_data(sdd, intents_perso)
model, words, classes, lemmatizer = model_training_load(data)

st.header("Box 1 de consultation, 9h")

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
            st.image(
                image_dict[st.session_state["generated"][i]], caption=None, width=190
            )
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

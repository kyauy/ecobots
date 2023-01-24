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

patient_descriptions = {
    "name": ["M. Denis"],
    "taille": ["Je fais 1m75"],
    "poids": ["Je fais 70kg"],
    "job": ["Je suis artiste peintre. J'aime mon métier."],
    "motif": [
        "Je viens vous consulter parce que j'ai peur d'avoir la maladie de Parkinson..."
    ],
    "stress": [
        "J'ai peur d'avoir la maladie de Parkinson...",
    ],
    "age": ["J'ai 55 ans"],
    "antecedant": ["J'ai de l'hypertension. Je prends du CAPTOPRIL le matin."],
    "antecedantfam": ["Pas à ma connaissance", "Je ne crois pas"],
    "howareyou": [
        "Ca va Docteur.",
        "Un peu stressée mais ca va.",
    ],
    "tabac": ["Non Docteur."],
}

sdd = [
    {
        "tag": "annonce",
        "patterns": [
            "Vous avez bien fait, nous allons prendre le temps de vous expliquer la maladie de Parkinson."
            "En effet, vous avez des symptomes pouvant faire penser à une maladie de Parkinson",
            "A l'examen, je retrouve des elements pouvant en effet indiquer que vous êtes atteint de la maladie de Parkison.",
        ],
        "responses": [
            "Je vous ecoute Docteur.",
            "Dites moi en plus Docteur s'il vous plait.",
        ],
    },
    {
        "tag": "explorationParkinson",
        "patterns": [
            "Vous présentez des tremblements, une lenteur aux mouvements et une rigidité qui peut faire penser à une maladie de Parkinson",
            "A l'examen, je retrouve des elements cliniques, comme des tremblements et une rigidité de vos mouvements. ",
        ],
        "responses": ["Mais vous êtes sur ? Y'a pas besoin d'examen complémentaires ?"],
    },
    {
        "tag": "confirmationParkinson",
        "patterns": [
            "La trisomie 21 est une maladie génétique qui associe des signes physiques et une atteinte neuro-neurodéveloppementale pour lequel une prise en charge précoce permet de mieux les accompagner.",
            "C'est une maladie grave",
        ],
        "responses": ["Est-ce que c'est grave ?"],
    },
    {
        "tag": "pronosticParkinson",
        "patterns": [
            "Nous devons faire une prise de sang, qui va rechercher la trisomie 21.",
            "Il s'agit d'une prise de sang",
            "Une prise de sang pour dépister la trisomie 21.",
            "Nous pouvons vous proposer une prise de sang",
            "Nous pouvons vous proposer un depistage non invasif par analyse de l'ADN libre circulant.",
            "nous pouvons vous proposer une nouvelle prise de sang pour être sur",
            "Il faudra réaliser une autre prise de sang pour DPNI, dépistage prénatal non invasif",
            "Il faut d'abord réaliser la prise de sang pour le DPNI",
        ],
        "responses": ["Est-ce que je peux vivre normalement ?"],
    },
    {
        "tag": "traitementParkinson",
        "patterns": [
            "Si le test est négatif, le suivi de la grossesse est normal. Si le doute persiste, nous devrons faire une amniocentèse pour avoir le diagnostic",
            "Si le test est positif, nous devrons faire une amniocentese pour determiner le diagnostic. C'est à dire prélever un peu de liquide amniotique.",
            "nous devrons faire une amniocentèse pour confirmer le diagnostic de trisomie 21.",
            "On va devoir confirmer cette suspicion, notamment graçe à une amniocentèse",
            "Il faut réaliser une amniocentèse. Savez-vous ce que c'est ?",
            "Si le DPNI est négatif, le risque de trisomie 21 sera très faible. S'il est positif, vous pourrez choisir de réaliser une biopsie de trophoblaste ou une amniocentèse pour confirmer la suspicion de trisomie 21 par la réalisation du caryotype foetal.",
            "Une prise de sang pour exclure la trisomie 21. si c'est positif il faudra faire une amniocentese.",
        ],
        "responses": ["Qu'est ce que vous me proposer ?"],
    },
    {
        "tag": "interactionParkinson",
        "patterns": [
            "C'est un examen fait en routine qui va recupérer du liquide amniotique pour faire une recherche génétique. Le risque de fausse couche est de 1/100.",
            "Le risque de fausse couche est de 1/100.",
            "C'est un examen fait en routine qui va recupérer du liquide amniotique pour faire une recherche génétique de la trisomie 21.",
            "Cela consiste à prélever du liquide amniotique pour pouvoir confirmer le diagnostic",
            "Il s'agit de prélever du liquide amniotique, il y a 1% de risque de perte foetale",
            "Il s'agit de prélever du liquide amniotique. Il y a 1% de risque de perte votre enfant.",
            "il s'agit d'un prelevement de liquide amniotique, il existe certains effets secondaire mais qui sont largement en dessous du bénéfice que nous procure cet examen.",
            "C'est la réalisation d'une ponction de liquide amniotique dans l'utérus. Ce n'est pas dangereux pour vous mais il y a un risque de fausse-couche.",
            "L'amniocentèse est un prélèvement du liquide amniotique, dans la poche qui entoure le foetus. Ce geste se réalise avec une aiguille par le ventre ou par voie vaginale. Il y a un risque de fausse couche induite inférieur à 1%.",
            "Que feriez-vous si votre enfant est atteint de trisomie 21?",
        ],
        "responses": [
            "Je prend un traitement pour la tension, est-ce qu'il y a un risque ?	"
        ],
    },
    {
        "tag": "reflexionParkinson",
        "patterns": [
            "Il faut prendre en charge précocement les complications médicales et débuter rapidement les rééducations pour l'accompagner aux mieux afin d'éviter le sur-handicap.",
            "Nous l'aiderons et rechercher les principales complication et les traiter afin d’éviter en particulier le sur-handicap. Il aura une marge de progression et la majorité des patients ont une certaine autonomie.",
            "Il sera accompagné et stimulé dans son enfance avec de la kiné, de l'orthophonie, de la psychomotricité, de l'ergothérapie, afin de lui permettre d'avoir la meilleure autonomie possible. Nous surveillerons les complications qui pourraient survenir, il et vous serez accompagné.",
            "une rééducation un dépistage des symptômes, une prise en charge personnalisée en fonction de ses besoins",
            "Un suivie rapproché sera nécessaire, notamment le long de la grossesse est a posteriori afin de deceler des complications en rapport avec cette pathologie.",
            "Un accompagnement vous sera proposer afin d'accueillir du mieux possible votre enfant.",
            "Si vous souhaitez le garder, et qu'il est atteint de trisomie 21, il y aura à la naissance une hypotonie, puis un retard des acquisitions qui évoluera vers une déficience intellectuelle , dont le degré est très variable selon les individus. Il s'agit d'une maladie qui touche plusieurs organes, avec un risque de cardiopathie congénitale, de troubles visuels, de troubles auditifs, de malformations viscérales, d'épilepsie... ",
            "Vous avez tout à fait le droit de garder votre enfant, si c'est le cas il sera suivi par des spécialistes après la naissance.",
            "Il aura une prise en charge multidisciplinaire avec un suivi régulier pour que tout se passe au mieux.",
        ],
        "responses": [
            "Je vais prendre le temps de réflechir avec vos explications avant de faire le test."
        ],
    },
]

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

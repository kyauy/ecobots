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
    # initial_sidebar_state="expanded",
)

nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

image_pg = Image.open("img/ecobots.png")
st.sidebar.image(image_pg, caption=None, width=100)
st.sidebar.header("ECOS Chatbot: Patient simulé par l'intelligence artificielle")
st.sidebar.markdown(
    """
**Lisez l'énoncé et lancez vous en disant bonjour à votre patient(e) !**

ECOS proposé par Matthieu Villessot et Kévin Yauy.  

Contact: [m-villessot@chu-montpellier.fr](mailto:m-villessot@chu-montpellier.fr) & [kevin.yauy@chu-montpellier.fr](mailto:kevin.yauy@chu-montpellier.fr)

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
def get_one_time_list_load(sdd, exception_list):
    return get_one_time_list(sdd, exception_list)


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

image_dict_sdd = {
    "Je vais prendre le temps de réflechir avec vos explications Docteur.": "img/whatsapp_fin.png",
}

image_dict.update(image_dict_sdd)

patient_descriptions = {
    "name": ["M. Denis"],
    "taille": ["Je fais 1m75"],
    "poids": ["Je fais 70kg"],
    "job": ["Je suis artiste peintre. J'aime mon métier."],
    "motif": [
        "Je viens vous consulter parce que j'ai peur d'avoir la maladie de Parkinson..."
    ],
    "stress": ["J'ai peur d'avoir la maladie de Parkinson.."],
    "age": ["J'ai 55 ans"],
    "antecedant": ["J'ai de l'hypertension. Je prends de l'amlodipine le matin."],
    "antecedantfam": ["Pas à ma connaissance", "Je ne crois pas"],
    "howareyou": ["Ca va Docteur.", "Un peu stressée mais ca va."],
    "tabac": ["Non Docteur."],
}

sdd = [
    {
        "tag": "suspicion",
        "patterns": [
            "Pourquoi pensez vous avoir la maladie de parkinson ?",
            "Qu'est ce qui vous fait penser que vous avez la maladie de Parkinson ?",
            "Qu'est ce qui vous fait pensez que vous avez une maladie de PArkinson ,",
            "pourquoi pensez vous avoir une maladie de Parkinson  ?",
        ],
        "responses": ["Je n'arrive pas à arreter de trembler du bras droit..."],
    },
    {
        "tag": "annonce",
        "patterns": [
            "Vous avez bien fait, nous allons prendre le temps de vous expliquer la maladie de Parkinson.",
            "En effet, vous avez des symptomes pouvant faire penser à une maladie de Parkinson",
            "A l'examen, je retrouve des elements pouvant en effet indiquer que vous êtes atteint de la maladie de Parkison.",
            "Les symptômes que vous présentez sont évocateur de syndrome parkinsonien, dont la maladie la plus représentative est la maladie de Parkinson.",
        ],
        "responses": ["Dites moi en plus Docteur s'il vous plait."],
    },
    {
        "tag": "explorationParkinson",
        "patterns": [
            "Vous présentez des tremblements, une lenteur aux mouvements et une rigidité qui peut faire penser à une maladie de Parkinson. ",
            "A l'examen, je retrouve des elements cliniques, comme des tremblements et une rigidité de vos mouvements.",
            "Alors les symptomes que vous présentez, le tremblement de repos, la lenteur dans vos mouvements me font penser à une maladie de parkison\n",
            "L'examen clinique montre un tremblement de repos, une rigidité, une lenteur évocatrice de maladie de PArkinson\n",
            "Alors les symptomes, les signes cliniques que vous présentez, notamment le tremblement de repos, la rigidité, la lenteur font évoquer une maladie de PARKINSON.",
        ],
        "responses": ["Qu'est ce qui cause la maladie de Parkinson?"],
    },
    {
        "tag": "explicationParkinson",
        "patterns": [
            "il s'agit d'une maladie qui est due à un déficit d'une molécule nommé dopamine dans le cerveau.",
            "Il s'agit d'une maladie qui est due à un déficit d'une molécule nommé dopamine dans le cerveau",
            "C'est une maladie neurodégénérative qui entraine un manque de dopamine.",
            "Il s'agit un déficit en dopamine dans le cerveau\n",
        ],
        "responses": ["Mais vous êtes sur ? Y'a pas besoin d'examen complémentaires ?"],
    },
    {
        "tag": "confirmationParkinson",
        "patterns": [
            "Tout à fait, le diagnostic est uniquement clinique. Il n'y a pas besoin de réaliser d'examen complémentaire. Le diagnostic de certitude est réalisé avec le suivi de l'absence d'autres signes pour un autre syndrome parkinsonien. Nous pourrons vous proposer un taritement dont la réponse nous orientera également sur le diagnostic.",
            "Tout à fait, le diagnostic est uniquement clinique. Il n'y a pas besoin de réaliser d'examen complémentaire. ",
            "Il n'y a pas besoin de réaliser d'examen complémentaire. Le diagnostic de certitude est réalisé avec le suivi de l'absence d'autres signes pour un autre syndrome parkinsonien. Nous pourrons vous proposer un taritement dont la réponse nous orientera également sur le diagnostic.",
            "Le diagnostic de maladie de Parkinson est un diagnostic clinique, mais seul le suivi et la réponse au traitement permettront d'affirmer le diagnostic\n",
            "Non le diagnostic est clinique mais c'est le suivi, votre réponse au traitement qui permettront d'affirmer le diagnostic\n",
        ],
        "responses": ["Est-ce que c'est grave ? Est ce qu'on peut en guérir ?"],
    },
    {
        "tag": "pronosticParkinson",
        "patterns": [
            "Il s'agit d'une maladie dont l'evolution propre à chaque individu. Si on ne peut pas la guérir, on peut vous proposer des traitements pour essayer de réduire vos symptomes.",
            "Il s'agit d'une maladie dont l'evolution propre à chaque individu. Si on ne peut pas la guérir, on peut vous proposer des traitements pour essayer de réduire vos tremblements et vos lenteurs de mouvements.",
            "Non ce n'est pas une maladie grave, mais chronique qui pourra avoir un retentissement sur votre quotidien C'est l'objectif du traitement de limiter son impact sur votre quotidien, avec une très grande efficacité initiale",
            "Non ce n'est pas une maladie curable, mais l'évolution est propre à chacun, c'est le suivi qui nous permettra de dire si cette maladie est invalidante ou non chez vous, et comment s'adapter au mieux",
        ],
        "responses": ["Est-ce que je peux vivre normalement ?"],
    },
    {
        "tag": "traitementParkinson",
        "patterns": [
            "On a des traitements qui peuvent soulager vos symptomes.",
            "On a des traitements symptomatiques.",
            "Selon la gène causée par vos tremblements ou mouvements, je peux vous proposer des traitements pour soulager vos symptomes. Ces traitements ne permettent pas de guerir mais peuvent aider à calmer vos symptomes.",
            "S'il n'existe pas de traitement pour guerir, il existe des traitements pour éviter vos symptomes, qui essaye de combler le déficit en dopamine qu'on retrouve dans la maladie de Parkinson.",
        ],
        "responses": ["Quel type de traitement existe? "],
    },
    {
        "tag": "typetraitementParkinson",
        "patterns": [
            "S'il n'existe pas de traitement pour guerir, il existe des traitements pour éviter vos symptomes, qui essaye de combler le déficit en dopamine qu'on retrouve dans la maladie de Parkinson. Ces traitement sont des agonistes dopaminergiques comme la Levodopa. Ce traitement peut avoir des effets secondaires.",
            "Selon la gène causée par vos tremblements ou mouvements, je peux vous proposer des traitements pour soulager vos symptomes. Ces traitements ne permettent pas de guerir mais peuvent aider à calmer vos symptomes. Ils essayent de combler le déficit en dopamine, comme la levodopa qui est un agoniste dopaminergique. Ce traitement peut avoir des effets secondaires.",
            "Alors on va introduire un traitement de vos symptomes pour que vous puissiez vivre le plus normalement possible, mais cela dépendra de votre réponse au traitement On a plusieurs types de traitement notamment la lévodopa et les agonistes dopaminergiques qui peuvent marcher sur vos symptomes",
            "nous pouvons vous proposer un traitement par dopamine",
        ],
        "responses": [
            "Je prend un traitement pour la tension, est-ce qu'il y a un risque ?"
        ],
    },
    {
        "tag": "interactionParkinson",
        "patterns": [
            "En effet, il y a un risque d'hypotension de l'association traitement anti-parkinsonien et Amlor.",
            "Alors non il n'y a pas de risque majeur mais le traitement de la maladie de parkinson peut faire baisser la tension et il faudra etre vigilant à ce sujet.",
            "non aucun risque",
            "Oui il y a un risque d'hypotension.",
        ],
        "responses": [
            "Je vais prendre le temps de réflechir avec vos explications Docteur."
        ],
    },
]

exception_list = []

# Generation des messages et du modèle d'IA

one_time_list = get_one_time_list_load(sdd, exception_list)
intents = get_intents()
intents_perso = personalize_intents_load(intents, patient_descriptions)
data = get_data(sdd, intents_perso)
model, words, classes, lemmatizer = model_training_load(data)

st.header("Box 1 de consultation, 9h")

(
    tab1,
    tab2,
    tab3,
) = st.tabs(["📝 Fiche Etudiant", "🕑 Commencez l'ECOS", "✅ Correction"])

with tab1:

    st.subheader("Contexte")
    st.markdown(
        """
Vous êtes interne en neurologie.

Vous voyez en consultation Mr Denis, 55 ans, qui vient vous voir pour tremblement de repos du bras droit, ralentissement dans l'execution de ses gestes. 
L'examen clinique vous révèle un syndrome parkinsonien asymétrique typique d'une maladie de Parkinson. 

Vous posez le diagnostic de maladie de Parkinson. 
Il se présente seul à votre consultation.
    """
    )
    st.subheader("Objectifs")
    st.markdown(
        """
- Vous annoncez au patient le diagnostic de maladie de Parkinson, et les principaux signes sur lequel il repose
- Vous en expliquez schématiquement la prise en charge thérapeutique.
- A la fin de la station, vous lui proposez un traitement de 1re intention.
    """
    )

    st.subheader("Prêt ?")
    st.markdown(
        """
    Cliquez sur la page "🕑 Commencez l'ECOS" !
    """
    )

    st.subheader("Debriefing et corrections")
    st.markdown(
        """
    Cliquez sur la page "✅ Correction" après avoir fait l'ECOS !
    """
    )


with tab2:

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
            message(
                st.session_state["generated"][i], key=str(i), avatar_style="pixel-art"
            )
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
with tab3:
    if st.session_state["generated"]:
        df = pd.DataFrame(
            list(zip(st.session_state["past"], st.session_state["generated"]))
        )
        df.columns = ["Vous", "Votre patient·e"]
        tsv = df.drop_duplicates().to_csv(sep="\t", index=False)
        st.download_button(
            label="Cliquez ici pour télécharger votre conversation",
            data=tsv,
            file_name="conversation_mv_sdd328_i106.tsv",
            mime="text/tsv",
        )
    st.markdown(
        """
        **Téléchargez la conversation et envoyez la via ce google form :**
        > [https://forms.gle/dVspfgx88k2HNEci9](https://forms.gle/dVspfgx88k2HNEci9)

        Vous receverez automatiquement le lien de la grille d'évaluation !
        """
    )

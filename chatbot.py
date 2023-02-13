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
    page_title="ECOS SDD 307 - Mme Miki",
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

ECOS proposé par Kévin Yauy.  

Contact: [kevin.yauy@chu-montpellier.fr](mailto:kevin.yauy@chu-montpellier.fr)

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
        "Vous (interne de gynéco-obstétrique): ",
        key="input",
        help="Discutez avec votre patiente avec des phrases complètes. Si problème: contactez kevin.yauy@chu-montpellier.fr",
        on_change=submit,
    )
    return st.session_state.answer


# JSON INPUT

## utilisation d'un dictionnaire pour représenter un fichier JSON d'intentions

image_dict_sdd = {
    "Je vais prendre le temps de réflechir avec vos explications avant de faire le test.": "img/whatsapp_fin.png",
}

image_dict.update(image_dict_sdd)

patient_descriptions = {
    "name": ["Mme Miki"],
    "taille": ["Je fais 1m65"],
    "poids": ["Je fais 70kg"],
    "job": ["Je suis animatrice de temps périscolaire. J'aime mon métier."],
    "motif": ["Je viens vous consulter pour avoir le résultat des prises de sang..."],
    "stress": [
        "J'ai peur des résultats que vous allez m'annoncer",
        "J'attendais les résultats que vous allez m'annoncer.",
    ],
    "age": ["J'ai 25 ans"],
    "antecedant": ["Je n'ai jamais été malade."],
    "antecedantfam": ["Pas à ma connaissance", "Je ne crois pas"],
    "howareyou": [
        "Je suis assez stressée par le rendez vous",
        "Un peu stressée par le rendez vous",
    ],
    "tabac": ["Oui Docteur, à peu près 5 cigarettes par jour."],
}

sdd = [
    {
        "tag": "motifdoc",
        "patterns": [
            "Ne vous inquiétez pas, tous va TRES bien se passer !"
            "On se voit aujourd'hui car j'ai recu des résultats d'examens pour vous.",
            "Vous souvenez-vous des examens réalisés lors de la dernière consultation ?",
            "J'ai recu des resultats des analyses.",
            "je dois vous rendre les résultat du dépistage de la trisomie 21",
            "on se voit aujourd'hui pour parler du dépistage de la trisomie 21 que vous avez réalisé",
            "Nous nous revoyons pour vos résultats de prise de sang",
            "Vos résultats sont globalement normaux, hormis le dépistage combiné de la trisomie 21",
        ],
        "responses": ["Je vous ecoute Docteur.", "Dites moi en plus Docteur."],
    },
    {
        "tag": "pronoT21",
        "patterns": [
            "L'ensemble des examens est revenu normal, excepté un risque estimé de trisomie 21 fœtale est de 1/970 qu'il nous faut explorer.",
            "Les examens ont retrouvé un risque qu'il faut explorer davantage de trisomie 21.",
            "Les différents examens ont révélé que votre foetus présente un risque sur 970 d'être trisomique",
            "La dernière fois, nous avions réalisé un depistage de la trisomie 21. Ce depistage est revenu avec un risque modéré. Nous devons faire d'autres analyses pour exclure ce diagnostic.",
            "le résultat de la trisomie 21 n'est pas normal",
            "Votre enfant présente un risque d'être porteur de trisomie 21",
            "A partir des examens on a donc obtenu un risque estimé significatif qui nécéssite d'autres examens",
            "J'ai reçu le résultat de la prise de sang, il y a un risque sur 970 que votre foetus soit porteur d'une trisomie 21.",
            "Il est à 1/970, nous pouvons vous proposer des examens complémentaires pour préciser votre risque",
        ],
        "responses": ["Vous pouvez m'en dire plus sur la trisomie ?"],
    },
    {
        "tag": "confirmT21",
        "patterns": [
            "La trisomie 21 est une maladie génétique qui associe des signes physiques et une atteinte neuro-neurodéveloppementale pour lequel une prise en charge précoce permet de mieux les accompagner.",
            "C'est une maladie grave",
            "C'est la présence d'un troisième chromosome 21 qui entraine un syndrome polymalformatif avec déficience intellectuelle",
            "L'atteinte peut êre variable mais toujours avec une déficience mentale au moins modérée.",
            "L'atteinte peut êre variable mais toujours avec un handicap intellectuel.",
            "La trisomie 21 est une maladie très variable dans l'expression, mais ici il s'agit uniquement d'un risque et nous ne sommes pas sur.",
            "Il peut avoir une déficience intellectuelle au moins modérée, avec un handicap.",
            "Des malformations, un deficit attentionnel est possible, des infections, un retard de langage, des malformations cardiaques peuvent survenir.",
            "Il s'agit d'une anomalie du nombre de chormosomes, ici sur la paire 21, qui peut entrainer des malformations et des retentissements sur le long terme.",
        ],
        "responses": ["Que faire Docteur pour être sur ?"],
    },
    {
        "tag": "postDPNI",
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
        "responses": ["Qu'est ce qui va se passer par la suite ?"],
    },
    {
        "tag": "Amniocentese",
        "patterns": [
            "Si le test est négatif, le suivi de la grossesse est normal. Si le doute persiste, nous devrons faire une amniocentèse pour avoir le diagnostic",
            "Si le test est positif, nous devrons faire une amniocentese pour determiner le diagnostic. C'est à dire prélever un peu de liquide amniotique.",
            "nous devrons faire une amniocentèse pour confirmer le diagnostic de trisomie 21.",
            "On va devoir confirmer cette suspicion, notamment graçe à une amniocentèse",
            "Il faut réaliser une amniocentèse. Savez-vous ce que c'est ?",
            "Si le DPNI est négatif, le risque de trisomie 21 sera très faible. S'il est positif, vous pourrez choisir de réaliser une biopsie de trophoblaste ou une amniocentèse pour confirmer la suspicion de trisomie 21 par la réalisation du caryotype foetal.",
            "Une prise de sang pour exclure la trisomie 21. si c'est positif il faudra faire une amniocentese.",
        ],
        "responses": ["C'est quoi l'amniocentèse ? C'est dangereux ?"],
    },
    {
        "tag": "questIMG",
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
        "responses": ["Je ne suis pas sur de vouloir un enfant avec une trisomie..."],
    },
    {
        "tag": "devT21",
        "patterns": [
            "Si vous le souhaitez, une interruption médicale de grossesse serait possible, après discusssion avec mes collegues.",
            "Pensez vous a interompre la grossesse ?",
            "Si votre enfant devait etre porteur d'une trisomie 21, cela changerait il quelque chose pour vous ? Pour la poursuite de la grossesse ?",
            "si le résultat confirme la trisomie on peut accepter une interruption médicale de grossesse",
            "Si le diagnostic est positif, vous pouvez demander une interruption médical de grossesse",
            "En france, si un tel diagnostic est confirmé alors, si vous le souhaitez, vous pouvez demander une interruption médicale de grossesse",
            "Certaines femmes avec un foetus atteint de trisomie 21 choisissent d'arrêter la grossesse",
        ],
        "responses": ["Et si je veux garder mon enfant, que vas t'il se passer ?"],
    },
    {
        "tag": "reflexionIMG",
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

exception_list = []

# Generation des messages et du modèle d'IA

one_time_list = get_one_time_list_load(sdd, exception_list)
intents = get_intents()
data = get_data(sdd, intents)
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
Vous êtes interne de gynéco-obstétrique.  
Vous voyez en consultation une patiente pour suivi de grossesse.  

Lors de la précédente et première consultation, vous avez prescrit les examens recommandés du premier trimestre, incluant un depistage de la trisomie 21 que la patiente a souhaité faire.  
L'ensemble des examens est revenu normal, excepté un risque estimé de trisomie 21 fœtale est de 1/970.
    """
    )
    st.subheader("Objectifs")
    st.markdown(
        """
- Vous devez établir une conduite à tenir.
- Vous devez répondre aux attentes du patient.
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
            file_name="conversation_ky_sdd328_i106.tsv",
            mime="text/tsv",
        )
    st.markdown(
        """
        **Téléchargez la conversation et envoyez la via ce google form :** 
        > [https://forms.gle/aSiwV5V4SHimcHZM9](https://forms.gle/aSiwV5V4SHimcHZM9)

        Vous receverez automatiquement le lien de la grille d'évaluation !
        """
    )

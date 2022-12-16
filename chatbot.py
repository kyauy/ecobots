import streamlit as st
from streamlit.components.v1 import html
import time
import pandas as pd
from streamlit_chat import message
from PIL import Image
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="ECOS - Chatbot test", page_icon=":robot:")

image_pg = Image.open("img/ecobots.png")
st.sidebar.image(image_pg, caption=None, width=100)

st.sidebar.header("ECOS Chatbot: Patient simulé par l'intelligence artificielle")

st.sidebar.markdown(
    """
- *Contexte:*

Vous êtes interne de gynéco-obstétrique.
Vous voyez en consultation une patiente pour suivi de grossesse.

Lors de la précédente et première consultation, vous avez prescrit les examens recommandés du premier trimestre, incluant un depistage de la trisomie 21 que la patiente a souhaité faire. 
L'ensemble des examens est revenu normal, excepté un risque estimé de trisomie 21 fœtale est de 1/970.

- *Objectifs:*

Vous devez établir une conduite à tenir.
Vous devez répondre aux attentes du patient.

Contact: [kevin.yauy@chu-montpellier.fr](mailto:kevin.yauy@chu-montpellier.fr)

"""
)


image_univ = Image.open("img/logosfacmontpellier.png")
st.sidebar.image(image_univ, caption=None, width=190)

image_chu = Image.open("img/CHU-montpellier.png")
st.sidebar.image(image_chu, caption=None, width=95)


# JSON INPUT

# utilisation d'un dictionnaire pour représenter un fichier JSON d'intentions
one_time_list = [
    "pronoT21",
    "devT21",
    "confirmT21",
    "postDPNI",
    "Amniocentese",
    "questIMG",
    "reflexionIMG",
]
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hello",
                "Bonjour",
                "Bonjour Mme MIKI",
                "Bonjour Madame",
                "Bonjour, je suis interne en Gynécologie",
                "Bonjour, je suis interne de gynéco-obstétrique",
                "Bonjour, je m'appelle , je suis interne de gynéco",
            ],
            "responses": ["Bonjour Docteur"],
        },
        {
            "tag": "taille",
            "patterns": ["Combien mesurez vous?", "Quel est votre taille?"],
            "responses": ["Je fais 1m65"],
        },
        {
            "tag": "poids",
            "patterns": ["Combien pesez vous?", "Quel est votre poids?"],
            "responses": ["Je fais 70kg"],
        },
        # {"tag": "job",
        # "patterns": ["Que faites vous dans la vie?", "Quelle est votre profession?"],
        # "responses": ["Je suis animatrice de temps périscolaire. J'aime mon métier."]
        # },
        {
            "tag": "interrogation",
            "patterns": [
                "Pourquoi?",
                "",
                "c'est un test",
                "c'est un test ?",
                "comment vous dire?",
                "IMG",
                "je suis perdu",
                "c'est difficile à dire",
                "les resultats",
                "bouh",
                "vous avez raison.",
                "je comprends que vous soyez stressée par vos résultats",
                "on se voit aujourd'hui",
                "Que vous a donc fait Mme MIKI ?",
            ],
            "responses": [
                "Que voulez vous dire Docteur ?",
                "C'est à dire Docteur ?",
                "Je ne comprends pas Docteur ?",
                "Vous pouvez répéter Docteur ?",
                "Vous pouvez répéter Docteur ?",
            ],
        },
        {
            "tag": "age",
            "patterns": ["Quel âge avez vous?", "Vous avez quel age ?"],
            "responses": ["J'ai 25 ans"],
        },
        {
            "tag": "antecedant",
            "patterns": ["Avez vous des maladies ?", "Avez vous déjà été opéré ?"],
            "responses": ["Je n'ai jamais été malade."],
        },
        #  {"tag": "antecedantfam",
        #  "patterns": ["Il y a t'il des membres de votre famille qui sont malades?", "Des maladies dans votre famille?"],
        #  "responses": ["Pas à ma connaissance", "Je ne crois pas"]
        # },
        # {"tag": "allergie",
        # "patterns": ["Avez vous des allergies?", "Etes vous allergique à quelque chose?"],
        # "responses": ["A la pénicilline, docteur. J'ai eu plein de boutons."]
        # },
        {
            "tag": "tabac",
            "patterns": ["Est ce que vous fumez?"],
            "responses": ["Oui Docteur, à peu près 5 cigarettes par jour."],
        },
        {
            "tag": "stress",
            "patterns": [
                "Bonjour, comment allez vous ?",
                "Bonjour madame, comment allez vous ?",
                "Comment allez vous?",
                "Comment ca va?",
                "Comment allez-vous?",
                "ca va ?",
                "Vous sentez-vous anxieuse concernant cette prise en charge ?",
            ],
            "responses": [
                "Je suis assez stressée par le rendez vous",
                "Un peu stressée par le rendez vous",
            ],
        },
        {
            "tag": "stress2",
            "patterns": [
                "Pourquoi êtes vous stressée ?",
                "Qu'est ce qui vous stresse ?",
            ],
            "responses": [
                "J'ai peur des résultats que vous allez m'annoncer",
                "J'attendais les résultats que vous allez m'annoncer.",
            ],
        },
        {
            "tag": "name",
            "patterns": ["Quel est votre nom?", "Comment vous vous appelez?"],
            "responses": ["Mme Miki"],
        },
        {
            "tag": "motif",
            "patterns": [
                "Qu'est ce qui vous amene?",
                "Savez vous pourquoi on se voit aujourd'hui ?",
                "Vous vous souvenez pourquoi nous avons rendez-vous aujourd'hui?",
                "Quel est votre problème ?",
                "Vous venez pour vos résultats ?",
            ],
            "responses": [
                "Je viens vous consulter pour avoir le résultat des prises de sang..."
            ],
        },
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
            "responses": [
                "Je ne suis pas sur de vouloir un enfant avec une trisomie..."
            ],
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
        {
            "tag": "agree",
            "patterns": [
                "Vous avez jusqu'à la fin de la grossesse pour y réfléchir",
            ],
            "responses": ["Entendu Docteur."],
        },
        {
            "tag": "auscultation",
            "patterns": [
                "Puis je vous ausculter votre coeur?",
                "Puis je vous ausculter?",
                "Pouvez vous enlever votre haut pour que je puisse vous ausculter?",
            ],
            "responses": ["Bien sur, auscultez moi Docteur."],
        },
        {
            "tag": "cartevitale",
            "patterns": [
                "Avez-vous la carte vitale ?",
                "pouvez vous me donner votre carte vitale ?",
                "J'aurai besoin de votre carte vitale, s'il vous plait ?",
                "J'aurai besoin de votre carte vitale.",
            ],
            "responses": ["La voici Docteur."],
        },
        {
            "tag": "dontknow",
            "patterns": [
                "Connaissez-vous le DPNI ?",
                "Connaissez-vous le dépistage non invastif de la trisomie 21 ?",
                "Connaissez-vous la trisomie 21 ?",
            ],
            "responses": ["Pas vraiment Docteur."],
        },
        {
            "tag": "thanks",
            "patterns": [
                "Vous pouvez toujours me joindre par mail, je vous tiendrai au courant des resultats de l'amniocentèse lors d'une prochaine consultation, afin de discuter de la suite de la prise en charge.",
                "Vous pouvez toujours me joindre par mail, je vous tiendrai au courant des resultats de la prise de sang lors d'une prochaine consultation, afin de discuter de la suite de la prise en charge.",
                "Je reste joignable par téléphone ou par mail si vous avez des questions.",
            ],
            "responses": ["Merci Docteur."],
        },
        {
            "tag": "question",
            "patterns": [
                "Avez vous encore des questions ?",
                "Avez vous des interrogations ?",
                "Voulez-vous un petit remontant ?",
                "Un petit verre de rouge ?",
                "Ben justement il serait temps d'arrêter cette consultation, avez-vous des questions en particulier ?",
            ],
            "responses": ["Pas pour le moment Docteur."],
        },
        {
            "tag": "question2",
            "patterns": [
                "Avez vous bien compris ?",
                "Voulez vous qu'on reprenne quelque chose ?",
            ],
            "responses": ["C'est clair Docteur."],
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Au revoir",
                "Au revoir madame",
                "On se revoit bientot.",
                "A bientôt madame",
                "Ciao",
                "Bye bye",
                "A bientôt",
            ],
            "responses": ["Merci. Au revoir Docteur."],
        },
    ]
}


# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://media.wbur.org/wp/2020/06/doctor-office-1000x667.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
# add_bg_from_url()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_training():
    # TRAIN THE MODEL
    # st.write('lemm')
    # initialisation de lemmatizer pour obtenir la racine des mots
    lemmatizer = WordNetLemmatizer()

    # création des listes
    words = []
    classes = []
    doc_X = []
    doc_y = []
    # st.write('intent')

    # parcourir avec une boucle For toutes les intentions
    # tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
    # le tag associé à l'intention sont ajoutés aux listes correspondantes
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_y.append(intent["tag"])

        # ajouter le tag aux classes s'il n'est pas déjà là
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    # lemmatiser tous les mots du vocabulaire et les convertir en minuscule
    # si les mots n'apparaissent pas dans la ponctuation
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word not in string.punctuation
    ]

    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
    words = sorted(set(words))
    classes = sorted(set(classes))

    # liste pour les données d'entraînement
    training = []
    out_empty = [0] * len(classes)
    # st.write('words')

    # création du modèle d'ensemble de mots

    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            if word in text:
                bow.append(1)
            else:
                bow.append(0)

        # marque l'index de la classe à laquelle le pattern atguel est associé à
        output_row = list(out_empty)
        output_row[classes.index(doc_y[idx])] = 1

        # ajoute le one hot encoded BoW et les classes associées à la liste training
        training.append([bow, output_row])

    # mélanger les données et les convertir en array
    random.shuffle(training)
    training = np.array(training, dtype=object)

    # séparer les features et les labels target
    train_X = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # définition de quelques paramètres
    input_shape = (len(train_X[0]),)
    output_shape = len(train_y[0])
    # st.write('deep config')

    # modèle Deep Learning
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation="softmax"))

    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    # st.write('compile')

    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    # st.write('fit')
    # entraînement du modèle
    model.fit(x=train_X, y=train_y, epochs=200, verbose=0)
    return model, words, classes, lemmatizer


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]), verbose=0)[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    return_list_comp = []
    for r in y_pred:
        return_list.append(labels[r[0]])
        return_list_comp.append({"response": labels[r[0]], "confidence": float(r[1])})
    return return_list, return_list_comp


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def query(user_input, debug):
    intents, return_list_comp = pred_class(user_input.lower(), words, classes)
    if debug == "On":
        st.write(pd.DataFrame(return_list_comp))
    if return_list_comp[0]["confidence"] < 0.6:
        result = "Pouvez vous répéter Docteur? je n'ai pas bien saisi."
    elif return_list_comp[0]["response"] in st.session_state["one_time_intent"]:
        result = "Je ne sais pas quoi dire Docteur."
    else:
        result = get_response(intents, data)
        if return_list_comp[0]["response"] in one_time_list:
            st.session_state.one_time_intent.append(return_list_comp[0]["response"])
    return result


def submit():
    st.session_state.answer = st.session_state.input
    st.session_state.input = ""


def get_text():
    input_text = st.text_input(
        "Vous (interne de gynéco-obstétrique): ",
        key="input",
        help="Discutez avec votre patiente avec des phrases complètes. Si problème: contactez kevin.yauy@chu-montpellier.fr",
        on_change=submit,
    )
    return st.session_state.answer


st.header("Box 4 de consultation, 9h30.")
# st.markdown("[Github](https://github.com/ai-yash/st-chat)"

model, words, classes, lemmatizer = model_training()

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

my_html = """
<script>
function startTimer(duration, display) {
    var timer = duration, minutes, seconds;
    setInterval(function () {
        minutes = parseInt(timer / 60, 10)
        seconds = parseInt(timer % 60, 10);

        minutes = minutes < 10 ? "0" + minutes : minutes;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        display.textContent = minutes;

        if (--timer < 0) {
            timer = duration;
        }
    }, 1000);
}

window.onload = function () {
    var sevenMinutes = 60 * 10,
        display = document.querySelector('#time');
    startTimer(sevenMinutes, display);
};
</script>
<div style="font-family:sans serif" color="#262730" align="right">Votre consultation doit se terminer dans <span id="time">10</span> minutes.</div>
"""
# display.textContent = minutes + ":" + seconds;

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

image_dict = {"La voici Docteur.": "img/carte_vitale.jpg"}
sound_dict = {"Bien sur, auscultez moi Docteur.": "mp3/normal_heart.mp3"}

if user_input:
    output = query(user_input, debug)
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

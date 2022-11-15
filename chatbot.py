import streamlit as st
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

nltk.download('omw-1.4', quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(
    page_title="ECOS - Chatbot test",
    page_icon=":robot:"
)

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
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "La forme?", "yo", "Salut", "Bonjour", "Bonjour Mme MIKI", 'Bonjour Madame'],
              "responses": ["Bonjour Docteur"],
             },
             {"tag": "taille",
              "patterns": ["Combien mesurez vous?", "Quel est votre taille?"],
              "responses": ["Je fais 1m65"]
             },
             {"tag": "poids",
              "patterns": ["Combien pesez vous?", "Quel est votre poids?"],
              "responses": ["Je fais 70kg"]
             },
             #{"tag": "job",
             # "patterns": ["Que faites vous dans la vie?", "Quelle est votre profession?"],
             # "responses": ["Je suis animatrice de temps périscolaire. J'aime mon métier."]
             #},
             {"tag": "interrogation",
              "patterns": ["Pourquoi?", "", "c'est un test", "c'est un test ?", 'comment vous dire?', 'IMG', "je suis perdu", "c'est difficile à dire", "les resultats", "bouh"],
              "responses": ["Que voulez vous dire Docteur ?", "C'est à dire Docteur ?", "Je ne comprends pas Docteur ?", "C'est à dire Docteur ?"]
             },
             {"tag": "age",
              "patterns": ["Quel âge avez vous?", "Quand êtes-vous né?", "Quand etes vous né?", "Vous avez quel age ?"],
              "responses": ["J'ai 25 ans"]
             },
              {"tag": "antecedant",
              "patterns": ["Avez vous des maladies ?", "Avez vous déjà été opéré ?"],
              "responses": ["Je n'ai jamais été malade"]
             },
            #  {"tag": "antecedantfam",
            #  "patterns": ["Il y a t'il des membres de votre famille qui sont malades?", "Des maladies dans votre famille?"],
            #  "responses": ["Pas à ma connaissance", "Je ne crois pas"]
             #},
             # {"tag": "allergie",
             # "patterns": ["Avez vous des allergies?", "Etes vous allergique à quelque chose?"],
             # "responses": ["A la pénicilline, docteur. J'ai eu plein de boutons."]
             #},
              {"tag": "tabac",
              "patterns": ["Est ce que vous fumez?"],
              "responses": ["Oui Docteur, a peu près 5 cigarettes par jour."]
             },
             {"tag": "stress",
              "patterns": ["Comment allez vous?", "Comment ca va?", "Comment allez-vous?", "ca va ?"],
              "responses": ["Je suis assez stressé par le rendez vous", "Un peu stressé"]
             },
             {"tag": "stress2",
              "patterns": ["Pourquoi êtes vous stressée ?", "Qu'est ce qui vous stresse ?"],
              "responses": ["J'ai peur des résultats que vous allez m'annoncer", "J'attendais les résultats que vous allez m'annoncer."]
             },
             {"tag": "name",
              "patterns": ["Quel est votre nom?", "Comment vous vous appelez?"],
              "responses": ["Mme Miki"]
             },
             {"tag": "motif",
              "patterns": ["Qu'est ce qui vous amene?", "Dites moi."],
              "responses": ["Je viens vous consulter pour avoir le résultat des prises de sang..."]
             },
             {"tag": "motifdoc",
              "patterns": ["On se voit aujourd'hui car j'ai recu des résultats d'examens pour vous.", "J'ai recu des resultats des analyses."],
              "responses": ["Je vous ecoute Docteur."]
             },
             {"tag": "pronoT21",
              "patterns": ["L'ensemble des examens est revenu normal, excepté un risque estimé de trisomie 21 fœtale est de 1/970 qu'il nous faut explorer.", "Les examens ont retrouvé un risque qu'il faut explorer davantage de trisomie 21.", "La dernière fois, nous avions réalisé un depistage de la trisomie 21. Ce depistage est revenu avec un risque modéré. Nous devons faire d'autres analyses pour exclure ce diagnostic." ],
              "responses": ["C'est grave la trisomie ?", "Vous pouvez m'en dire plus sur la trisomie ?"]
             },
             {"tag": "confirmT21",
              "patterns": ["La trisomie 21 est une maladie génétique qui associe des signes physiques et une atteinte neuro-neurodéveloppementale pour lequel une prise en charge précoce permet de mieux les accompagner.", "C'est une maladie grave", "L'atteinte peut êre variable mais toujours avec une déficience mentale au moins modérée." , "L'atteinte peut êre variable mais toujours avec un handicap intellectuel.", "La trisomie 21 est une maladie très variable dans l'expression, mais ici il s'agit uniquement d'un risque et nous ne sommes pas sur." ,  "Il peut avoir une déficience intellectuelle au moins modérée, avec un handicap.", "Des malformations, un deficit attentionnel est possible, des infections, un retard de langage, des malformations cardiaques peuvent survenir."],
              "responses": ["Que faire Docteur pour être sur ?"]
             },
             {"tag": "postDPNI",
              "patterns": ["Nous devons faire une prise de sang, qui va rechercher la trisomie 21.", "Il s'agit d'une prise de sang", "Il s'agit d'une prise de sang", "Nous pouvons vous proposer un depistage non invasif par analyse de l'ADN libre circulant."],
              "responses": ["Qu'est ce qui va se passer par la suite ?"]
             },
             {"tag": "Amnicocentese",
              "patterns": ["Si le test est négatif, le suivi de la grossesse est normal. Si le doute persiste, nous devrons faire une amniocentèse pour avoir le diagnostic", "Si le test est positif, nous devrons faire une amniocentese pour determiner le diagnostic."],
              "responses": ["C'est quoi l'amniocentèse ? C'est dangereux ?"]
             },
             {"tag": "questIMG",
              "patterns": ["C'est un examen fait en routine qui va recupérer du liquide amniotique pour faire une recherche génétique. Le risque de fausse couche est de 1/100.", "Le risque de fausse couche est de 1/100.", "C'est un examen fait en routine qui va recupérer du liquide amniotique pour faire une recherche génétique de la trisomie 21." ],
              "responses": ["Je ne suis pas sur de vouloir un enfant avec une trisomie..."]
             },
            {"tag": "devT21",
              "patterns": ["Si vous le souhaitez, une interruption médicale de grossesse serait possible, après discusssion avec mes collegues.", "Pensez vous a interompre la grossesse ?", "Si votre enfant devait etre porteur d'une trisomie 21, cela changerait il quelque chose pour vous ? Pour la poursuite de la grossesse ?"],
              "responses": ["Si je veux garder mon enfant, que vas t'il se passer ?"]
             },
            {"tag": "reflexionIMG",
              "patterns": ["Il faut prendre en charge précocement les complications médicales et débuter rapidement les rééducations pour l'accompagner aux mieux afin d'éviter le sur-handicap.", "Nous l'aiderons et rechercher les principales complication et les traiter afin d’éviter en particulier le sur-handicap. Il aura une marge de progression et la majorité des patients ont une certaine autonomie.", "Il sera accompagné et stimulé dans son enfance avec de la kiné, de l'orthophonie, de la psychomotricité, de l'ergothérapie, afin de lui permettre d'avoir la meilleure autonomie possible. Nous surveillerons les complications qui pourraient survenir, il et vous serez accompagné."],
              "responses": ["Je vais prendre le temps de réflechir avec vos explications."]
             },
             {"tag": "question",
              "patterns": [ "Avez vous encore des questions ?", "Avez vous des interrogations ?"],
              "responses": ["Pas pour le moment Docteur"]
             },
             {"tag": "question2",
              "patterns": [ "Avez vous bien compris ?", "Voulez vous qu'on reprenne quelque chose ?"],
              "responses": ["C'est clair Docteur"]
             },
             {"tag": "goodbye",
              "patterns": [ "Au revoir", "Au revoir madame"],
              "responses": ["Merci Docteur."]
             }]
}
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_training():
    # TRAIN THE MODEL
    #st.write('lemm')
    # initialisation de lemmatizer pour obtenir la racine des mots
    lemmatizer = WordNetLemmatizer()

    # création des listes
    words = []
    classes = []
    doc_X = []
    doc_y = []
    #st.write('intent')

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
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
    words = sorted(set(words))
    classes = sorted(set(classes))

    # liste pour les données d'entraînement
    training = []
    out_empty = [0] * len(classes)
    #st.write('words')

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
    #st.write('deep config')

    # modèle Deep Learning
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation = "softmax"))

    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    #st.write('compile')

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    #st.write('fit')
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
    return_list_comp.append({"reponse": labels[r[0]], "confiance": float(r[1])})
  return return_list, return_list_comp

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

model, words, classes, lemmatizer = model_training()

st.header("Box 4 de consultation, 9h30.")
#st.markdown("[Github](https://github.com/ai-yash/st-chat)"


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(user_input, debug):
    intents, return_list_comp = pred_class(user_input.lower(), words, classes)
    if debug == "On":
      st.write(pd.DataFrame(return_list_comp))
    if return_list_comp[0]["confiance"] < 0.6:
      result = "Pouvez vous répéter Docteur? je n'ai pas bien saisi."
    else:
      result = get_response(intents, data) 
    return result

def get_text():
    input_text = st.text_input("Vous (interne de gynéco-obstétrique): ", key="input", help="Discutez avec votre patiente avec des phrases complètes. Si problème: contactez kevin.yauy@chu-montpellier.fr")
    return input_text 

if st.button('Debug mode'):
  debug = "On"
else:
  debug = "Off"

user_input = get_text()

if user_input:
    output = query(user_input, debug)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="pixel-art")
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="pixel-art-neutral")
    df = pd.DataFrame(list(zip(st.session_state['past'],st.session_state['generated'])))
    df.columns = ['Vous', 'Votre patient·e']
    tsv = df.drop_duplicates().to_csv(sep="\t", index=False)
    st.download_button(
        label="Téléchargez votre conversation",
        data=tsv,
        file_name='conversation.tsv',
        mime='text/tsv'
    )

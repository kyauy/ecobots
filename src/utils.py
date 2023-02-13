# coding=utf-8
# =========================================================================
# Modules
# =========================================================================
import streamlit as st
import random
import nltk
import pandas as pd
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def model_training(data):
    # st.write("Cache miss", data)
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


def clean_text(text, lemmatizer):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, lemmatizer, vocab):
    tokens = clean_text(text, lemmatizer)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, lemmatizer, labels, model):
    bow = bag_of_words(text, lemmatizer, vocab)
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


def query(user_input, debug, one_time_list, model, lemmatizer, words, classes, data):
    intents, return_list_comp = pred_class(
        user_input.lower(), words, lemmatizer, classes, model
    )
    if debug == "On":
        st.write(pd.DataFrame(return_list_comp))
    if return_list_comp[0]["confidence"] < 0.5:
        result = "Pouvez vous répéter Docteur? je n'ai pas bien saisi."
    elif return_list_comp[0]["response"] in st.session_state["one_time_intent"]:
        result = "Je ne sais pas quoi dire Docteur, nous en avons déjà parlé."
    else:
        result = get_response(intents, data)
        if return_list_comp[0]["response"] in one_time_list:
            st.session_state.one_time_intent.append(return_list_comp[0]["response"])
    return result


def submit():
    st.session_state.answer = st.session_state.input
    st.session_state.input = ""
    return ""


def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://media.wbur.org/wp/2020/06/doctor-office-1000x667.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


def get_one_time_list(sdd, exception_list):
    one_time_list = []
    for element in sdd:
        if element not in exception_list:
            one_time_list.append(element["tag"])
    return one_time_list


def get_data(sdd, intents):
    data = {"intents": []}
    for element in intents:
        data["intents"].append(element)
    for element in sdd:
        data["intents"].append(element)
    return data


def personalize_intents(intents, patients_descriptions):
    for element in intents:
        if element["tag"] in patients_descriptions.keys():
            element["responses"] = patients_descriptions[element["tag"]]
    return intents


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

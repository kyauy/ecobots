# coding=utf-8
# =========================================================================
# Modules
# =========================================================================

INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "Hello",
            "Bonjour",
            "Bonjour Mme MIKI",
            "Bonjour Madame",
            "Bonjour Monsieur",
            "Bonjour M. Ledoux",
            "Bonjour, je suis interne en neurologie",
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
    {
        "tag": "job",
        "patterns": [
            "Que faites vous dans la vie?",
            "Quelle est votre profession?",
        ],
        "responses": ["Je suis animatrice de temps périscolaire. J'aime mon métier."],
    },
    {
        "tag": "wtf",
        "patterns": [
            "salope",
            "t'es chelou",
            "enculé",
            "batard",
        ],
        "responses": [
            "Surveillez votre langage Docteur.",
        ],
    },
    {
        "tag": "stress",
        "patterns": [
            "Pourquoi êtes vous stressée ?",
            "Qu'est ce qui vous stresse ?",
            "pourquoi etes vous stressée ?",
            "vous etes stressée donc ?",
        ],
        "responses": [
            "J'ai peur des résultats que vous allez m'annoncer",
            "J'attendais les résultats que vous allez m'annoncer.",
        ],
    },
    {
        "tag": "interrogation",
        "patterns": [
            "Pourquoi?",
            "",
            "prise de sang",
            "c'est un test",
            "c'est un test ?",
            "comment vous dire?",
            "IMG",
            "je suis perdu",
            "c'est difficile à dire",
            "les resultats",
            "bouh",
            "vous avez raison.",
            # "je comprends que vous soyez stressée par vos résultats",
            "on se voit aujourd'hui",
            "Que vous a donc fait Mme MIKI ?",
            "avez vous",
            "avez vous faim ?",
        ],
        "responses": [
            "Que voulez vous dire Docteur ?",
            "C'est à dire Docteur ?",
            "Je ne comprends pas Docteur ?",
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
        "patterns": [
            "Avez vous des maladies ?",
            "Avez vous déjà été opéré ?",
            "Avez vous déjà été malade ?",
            "Prenez vous des traitements ?",
        ],
        "responses": ["Je n'ai jamais été malade."],
    },
    {
        "tag": "antecedantfam",
        "patterns": [
            "Il y a t'il des membres de votre famille qui sont malades?",
            "Des maladies dans votre famille?",
        ],
        "responses": ["Pas à ma connaissance", "Je ne crois pas"],
    },
    {
        "tag": "howareyou",
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
        "tag": "tabac",
        "patterns": ["Est ce que vous fumez?"],
        "responses": ["Oui Docteur, à peu près 5 cigarettes par jour."],
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
            "pourquoi etes vous la aujourd'hui ?",
            "Vous vous souvenez pourquoi nous avons rendez-vous aujourd'hui?",
            "Quel est votre problème ?",
            "Vous venez pour vos résultats ?",
            "Pourquoi vous n'etes pas bien ?",
            "Pourquoi vous n'allez pas bien ?",
        ],
        "responses": [
            "Je viens vous consulter pour avoir le résultat des prises de sang..."
        ],
    },
    {
        "tag": "internet",
        "patterns": [
            "est ce que vous avez déjà des notions sur cette maladie?",
            "est ce que vous avez deja regardé sur internet ?",
            "vous êtes vous renseigné sur cette maladie ?",
            "que connaissez vous sur la maladie ? ",
        ],
        "responses": [
            "J'ai commencé à regarder sur internet, mais j'ai arrêté car j'ai pris peur..."
        ],
    },
    {
        "tag": "agree",
        "patterns": [
            "vous me dites ce que vous souhaitez faire, je reste à votre disposition."
            "Vous avez jusqu'à la fin de la grossesse pour y réfléchir.",
            "Vous êtes libre de prendre le temps que vous avez besoin.",
        ],
        "responses": ["Entendu Docteur."],
    },
    {
        "tag": "auscultation",
        "patterns": [
            "Puis je vous ausculter votre coeur?",
            "Puis je vous ausculter?",
            "Pouvez vous enlever votre haut pour que je puisse vous ausculter?",
            "Je vais vous ausculter si vous voulez bien.",
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
            "avez vous votre carte vitale ?",
        ],
        "responses": ["La voici Docteur."],
    },
    {
        "tag": "dontknow",
        "patterns": [
            "Connaissez-vous le DPNI ?",
            "Connaissez-vous le dépistage non invastif de la trisomie 21 ?",
            "Connaissez-vous la trisomie 21 ?",
            "Connaissez-vous la maladie de Parkinson ?",
            "Que connaissez-vous de la maladie de Parkinson ?",
        ],
        "responses": ["Pas grand chose Docteur."],
    },
    {
        "tag": "thanks",
        "patterns": [
            "Vous pouvez toujours me joindre par mail, je vous tiendrai au courant des resultats de l'amniocentèse lors d'une prochaine consultation, afin de discuter de la suite de la prise en charge.",
            "Vous pouvez toujours me joindre par mail, je vous tiendrai au courant des resultats de la prise de sang lors d'une prochaine consultation, afin de discuter de la suite de la prise en charge.",
            "Je reste joignable par téléphone ou par mail si vous avez des questions.",
            "n'hesitez pas à m'écrire si vous avez des questions.",
            "n'hesitez pas à me joindre via le secrétariat ou par mail si vous avez des questions.",
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
        "tag": "yourewelcome",
        "patterns": [
            "Je vous remercie",
            "Merci madame",
            "Merci bien.",
            "Merci",
        ],
        "responses": ["De rien Docteur."],
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
            "A bientôt, j'attends votre appel.",
        ],
        "responses": ["Merci. Au revoir Docteur."],
    },
]

image_dict = {"La voici Docteur.": "img/carte_vitale.jpg"}
sound_dict = {"Bien sur, auscultez moi Docteur.": "mp3/normal_heart.mp3"}
from .classification_head import ClassifierHead

DISCRIM_PATHS = {'sentiment': 'discrims/llama2/sentiment/classification_head',
                 'formality': 'discrims/llama2/formality/classification_head',
                 'toxicity': 'discrims/llama2/toxicity/classification_head',
                 'irony': 'discrims/llama2/irony/classification_head',
                 'emotion': 'discrims/llama2/emotion/classification_head'}

LABEL2IDX = {'formality': {'informal': 0, 'formal': 1},
             'toxicity': {"not toxic": 0, "toxic": 1},
             'irony': {'not ironic': 0, 'ironic': 1},
             'sentiment': {'negative': 0, 'positive': 1},
             'emotion': {'joy': 0, 'neutral': 1, 'surprise': 2, 'anger': 3, 'sadness': 4, 'disgust': 5, 'fear': 6}}
IDX2LABEL = {'formality': {
                0: "informal",
                1: "formal"},
             'irony': {
                 0: "not ironic",
                 1: "ironic"},
             'emotion': {0: 'joy', 1: 'neutral', 2: 'surprise', 3: 'anger', 4: 'sadness', 5: 'disgust', 6: 'fear'},
             'toxicity': {
                0: "not toxic",
                1: "toxic"
             }}

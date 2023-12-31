from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
import json
import os
import pandas as pd
from nlp_model.pre_process import PreProcess
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences

def index(request):
    # if request.method == 'POST':
    #     model = load("D:\\django\\testDJ\\nlp\\nlp_model\\model_svm")
    #     json_object = json.loads(request.body.decode("utf-8"))
    #     emailJson = json_object["email"]
    #     email = pd.Series(data=[emailJson], index=[0])
    #     email = PreProcess.preProcess(email)
    #     email[0] = PreProcess.standardize(email[0])
    #     print("after preprocess: " + email)

    #     myVocabulary = load("D:\\django\\testDJ\\nlp\\nlp_model\\vocabulary")
    #     cv = CountVectorizer()
    #     cv.fit_transform(myVocabulary)
    #     features_CV = cv.transform(email)
    #     result = model.predict(features_CV)
    #     return HttpResponse(json.dumps({ "predict": result.item(0) }))
    
    model = load(os.path.abspath("./nlp_model/model_svm"))
    json_object = json.loads(request.body.decode("utf-8"))
    emailJson = json_object["email"]
    email = pd.Series(data=[emailJson], index=[0])
    email = PreProcess.preProcess(email)
    email[0] = PreProcess.standardize(email[0])

    myVocabulary = load(os.path.abspath("./nlp_model/vocabulary"))
    cv = CountVectorizer()
    cv.fit_transform(myVocabulary)
    features_CV = cv.transform(email)
    result = model.predict(features_CV)
    print(result.item(0))

    cnn_tokenizer = load(os.path.abspath("./nlp_model/cnn_tokenizer"))
    cnn_model = load(os.path.abspath("./nlp_model/model_cnn"))
    cnn_data = cnn_tokenizer.texts_to_sequences(email)
    cnn_data = pad_sequences(cnn_data, maxlen=40)
    cnn_prediction = cnn_model.predict(cnn_data)
    print(cnn_prediction.item(0))
    cnn_result = 0
    if (cnn_prediction.item(0) > 0.5):
        cnn_result = 1

    return HttpResponse(json.dumps({ "svm": result.item(0), "cnn": cnn_result }))

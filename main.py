import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from multiprocessing import Pool
import seaborn as sns
import os
from matplotlib import pyplot as plt
import pickle
import requests
import traceback
def send_to_telegram(message):

    apiToken = '5765471758:AAFPzn2Z2gbbe0sp6yurqxwbSmYrrGanla4'
    chatID = '1479006629'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)
class VotingClassifier:
    def __init__(self,models,weights=None,n_jobs=None):
        self.models = models
        if weights is None:
            self.weights = [1 for _ in range(len(self.models))]
        else:
            self.weights = weights
        self.n_jobs = n_jobs
    def predict(self,X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(X))
        predictions = np.array(predictions)
        for i in range(len(predictions)):
            predictions[i] = predictions[i]*self.weights[i]
        predictions = np.mean(predictions,axis=0)
        return np.argmax(predictions,axis=1)

    def score(self,X,y):
        y_pred = self.predict(X)
        return np.sum(y_pred==y)/len(y)

    def get_metrices(self,X,y):
        y_pred = self.predict(X)
        return classification_report(y,y_pred,output_dict=True),confusion_matrix(y,y_pred)
def save_model(model,pca_level,clf_name,BaseLanguage):
    with open(os.path.join("models",BaseLanguage+f"{clf_name}_pca{pca_level}.pkl"),'wb') as f:
        pickle.dump(model,f)

def getAccuracy(args):
    BaseLanguage,pca_level,svc_params,rf_params = args
    df = pd.read_csv(os.path.join("data",BaseLanguage+f"pca{pca_level}.csv"))
    y = df.iloc[:,0]
    X = df.iloc[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    svc = SVC(probability=True,kernel=svc_params["kernel"],C=svc_params["C"])
    svc.fit(X_train,y_train)
    svc_score = svc.score(X_test,y_test)
    svc_pred = svc.predict(X_test)
    svc_cnf = confusion_matrix(y_test,svc_pred)
    sns.heatmap(svc_cnf,annot=True,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"SVC Confusion Matrix {pca_level}")
    plt.savefig(os.path.join("confusion_matrix",BaseLanguage+f"svc_pca{pca_level}.png"))
    svc_clf = classification_report(y_test,svc_pred,output_dict=True)
    pd.DataFrame(svc_clf).transpose().to_csv(os.path.join("classification_report",BaseLanguage+f"svc_pca{pca_level}.csv"))
    print(f"{pca_level} SVC score {svc_score}")
    save_model(svc, pca_level, "svc",BaseLanguage)
    rf = RandomForestClassifier(n_estimators=rf_params["n_estimators"],max_depth=rf_params["max_depth"])
    rf.fit(X_train,y_train)
    rf_score = rf.score(X_test,y_test)
    rf_pred = rf.predict(X_test)
    rf_cnf = confusion_matrix(y_test,rf_pred)
    sns.heatmap(rf_cnf,annot=True,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"RF Confusion Matrix {pca_level}")
    plt.savefig(os.path.join("confusion_matrix",BaseLanguage+f"rf_pca{pca_level}.png"))
    rf_clf = classification_report(y_test,rf_pred,output_dict=True)
    pd.DataFrame(rf_clf).transpose().to_csv(os.path.join("classification_report",BaseLanguage+f"rf_pca{pca_level}.csv"))
    print(f"{pca_level} RF score {rf_score}")
    save_model(rf, pca_level, "rf",BaseLanguage)
    votingClf = VotingClassifier([svc,rf])
    votingClf_score = votingClf.score(X_test,y_test)
    votingClf_pred = votingClf.predict(X_test)
    votingClf_cnf = confusion_matrix(y_test,votingClf_pred)
    sns.heatmap(votingClf_cnf,annot=True,fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Voting Confusion Matrix {pca_level}")
    plt.savefig(os.path.join("confusion_matrix",BaseLanguage+f"voting_pca{pca_level}.png"))
    votingClf_clf = classification_report(y_test,votingClf_pred,output_dict=True)
    pd.DataFrame(votingClf_clf).transpose().to_csv(os.path.join("classification_report",BaseLanguage+f"voting_pca{pca_level}.csv"))
    print(f"{pca_level} voting score {votingClf_score}")
    save_model(votingClf, pca_level, "voting",BaseLanguage)
    send_to_telegram(f"{BaseLanguage} {pca_level} done with {votingClf_score}")
    return pca_level,svc_score,rf_score,votingClf_score

if __name__ == "__main__":
    datasets = ["Ravdess","German","Persian","Italian","Bangla"]
    BaseLanguages = [i+"_" for i in datasets]
    send_to_telegram("Started for languages\n"+("\n").join(datasets))
    for BaseLanguage in BaseLanguages:
        try:
            extensions = [f"pca{i}.csv" for i in [10,20,30,40,50]]
            basefile = os.path.join("data",BaseLanguage+"pca0.csv")
            f = open(BaseLanguage+"output.txt","w")
            df = pd.read_csv(basefile)
            y = df.iloc[:,0]
            X = df.iloc[:,1:]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
            svc_params = {"kernel":["linear","poly","sigmoid","rbf"],"C":[1,10,100]}
            svc = SVC(probability=True)
            svc_search = GridSearchCV(svc,svc_params,verbose=4,n_jobs=-1)
            svc_search.fit(X_train,y_train)
            svc_score = svc_search.score(X_test,y_test)
            svc_pred = svc_search.predict(X_test)
            svc_cnf = confusion_matrix(y_test,svc_pred)
            sns.heatmap(svc_cnf,annot=True,fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("SVC Base Confusion Matrix")
            plt.savefig(os.path.join("confusion_matrix",BaseLanguage+"svc_base.png"))
            svc_clf = classification_report(y_test, svc_pred, output_dict=True)
            pd.DataFrame(report).transpose().to_csv(os.path.join("classification_report",BaseLanguage+"svc_base.csv"))
            save_model(svc_search, 0, "svc",BaseLanguage)
            print(f"Base SVC score: {svc_score}")
            f.write(f"Base SVC score: {svc_score}"+"\n")
            rf_params = {"n_estimators":[30,40,50],"max_depth":[25,30,35,40]}
            rf = RandomForestClassifier()
            rf_search = GridSearchCV(rf,rf_params,verbose=4,n_jobs=-1)
            rf_search.fit(X_train,y_train)
            rf_score = rf_search.score(X_test,y_test)
            rf_pred = rf_search.predict(X_test)
            rf_cnf = confusion_matrix(y_test,rf_pred)
            sns.heatmap(rf_cnf,annot=True,fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("RF Base Confusion Matrix")
            plt.savefig(os.path.join("confusion_matrix",BaseLanguage+"rf_base.png"))
            rf_clf = classification_report(y_test, rf_pred,output_dict=True)
            pd.DataFrame(report).transpose().to_csv(os.path.join("classification_report",BaseLanguage+"rf_base.csv"))
            save_model(rf_search, 0, "rf",BaseLanguage)
            print(f"Base RF score: {rf_score}")
            f.write(f"Base RF score: {rf_score}"+"\n")
            votingclf = VotingClassifier([svc_search,rf_search])
            voting_score = votingclf.score(X_test,y_test)
            voting_pred = votingclf.predict(X_test)
            voting_cnf = confusion_matrix(y_test,voting_pred)
            sns.heatmap(voting_cnf,annot=True,fmt="d")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Voting Base Confusion Matrix")
            plt.savefig(os.path.join("confusion_matrix",BaseLanguage+"voting_base.png"))
            voting_clf = classification_report(y_test, voting_pred)
            pd.DataFrame(report).transpose().to_csv(os.path.join("classification_report",BaseLanguage+"voting_base.csv"))
            save_model(votingclf, 0, "voting",BaseLanguage)
            print(f"Base Voting score: {voting_score}")
            f.write(f"Base Voting score: {voting_score}"+"\n")
            send_to_telegram(f"Base Done for {BaseLanguage} with {voting_score}")
            args = [(BaseLanguage,pca_level,svc_search.best_params_,rf_search.best_params_) for pca_level in range(10,55,5)]
            args.append((BaseLanguage,52,svc_search.best_params_,rf_search.best_params_))
            print("Processing the PCA models")
            with Pool(4) as p:
                outputs = p.map(getAccuracy,args)
            for pca_level,svc_score,rf_score,votingClf_score in outputs:
                print(f"PCA level: {pca_level} SVC score: {svc_score} RF score: {rf_score} Voting score: {votingClf_score}")
                f.write(f"PCA level: {pca_level} SVC score: {svc_score} RF score: {rf_score} Voting score: {votingClf_score}"+"\n")
            f.close()
            send_to_telegram(f"{BaseLanguage} done")
        except Exception as e:
            print(e)
            traceback.print_exc()
            send_to_telegram(f"Error {str(e)}")

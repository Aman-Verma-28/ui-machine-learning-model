import os
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Random Forest
from sklearn.ensemble import RandomForestClassifier
import xgboost as Xgb
# Support Vector Machines
from sklearn.svm import SVC

for dirname, _, filenames in os.walk('/home/amanlinux/Downloads/Major Project/akarsh/akarsh/base/classData.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def predictor(ia, ib, ic, va, vb, vc):
    warnings.filterwarnings('ignore')
    sns.set_theme(context='notebook',
                  style='white',
                  palette='deep',
                  font='Lucida Calligraphy',
                  font_scale=1.5,
                  color_codes=True,
                  rc=None)
    plt.rcParams['figure.figsize'] = (14,8) 
    plt.rcParams['figure.facecolor'] = '#F0F8FF'
    plt.rcParams['figure.titlesize'] = 'medium'
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.edgecolor'] = 'green'
    plt.rcParams['figure.frameon'] = True
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['axes.facecolor'] = '#F5F5DC'
    plt.rcParams['axes.titlesize'] = 25   
    plt.rcParams["axes.titleweight"] = 'normal'
    plt.rcParams["axes.titlecolor"] = 'Olive'
    plt.rcParams['axes.edgecolor'] = 'pink'
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.titlelocation'] = 'center' 
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.labelpad"] = 2
    plt.rcParams['axes.labelweight'] = 1
    plt.rcParams["axes.labelcolor"] = 'Olive'
    plt.rcParams["axes.axisbelow"] = False 
    plt.rcParams['axes.xmargin'] = .2
    plt.rcParams["axes.ymargin"] = .2
    plt.rcParams["xtick.bottom"] = True 
    plt.rcParams['xtick.color'] = '#A52A2A'
    plt.rcParams["ytick.left"] = True  
    plt.rcParams['ytick.color'] = '#A52A2A'
    plt.rcParams['axes.grid'] = True 
    plt.rcParams['grid.color'] = 'green'
    plt.rcParams['grid.linestyle'] = '--' 
    plt.rcParams['grid.linewidth'] = .5
    plt.rcParams['grid.alpha'] = .3       
    plt.rcParams['legend.loc'] = 'best' 
    plt.rcParams['legend.facecolor'] =  'NavajoWhite'  
    plt.rcParams['legend.edgecolor'] = 'pink'
    plt.rcParams['legend.shadow'] = True
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['font.family'] = 'Lucida Calligraphy'
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.edgecolor'] = 'Blue'
    df_class = pd.read_csv("/home/amanlinux/Downloads/Major Project/akarsh/akarsh/base/classData.csv")
    df_class.sample(5).style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    df_class.info()
    df_class.shape
    ax = plt.subplot(1,2,1)
    ax = sns.countplot(x='G', data=df_class)
    plt.title("Ground Fault", fontsize=20,color = 'Brown',font='Lucida Calligraphy',pad=20)
    ax =plt.subplot(1,2,2)
    ax=df_class['G'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
    ax.set_title(label = "Ground Fault", fontsize = 20,color='Brown',font='Lucida Calligraphy',pad=20);
    ax = plt.subplot(1,2,1)
    ax = sns.countplot(x='A', data=df_class)
    plt.title("Line A Fault", fontsize=20,color = 'Brown',font='Lucida Calligraphy',pad=20)
    ax =plt.subplot(1,2,2)
    ax=df_class['A'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
    ax.set_title(label = "Line A Fault", fontsize = 20,color='Brown',font='Lucida Calligraphy',pad=20);
    ax = plt.subplot(1,2,1)
    ax = sns.countplot(x='B', data=df_class)
    plt.title("Line B Fault", fontsize=20,color = 'Brown',font='Lucida Calligraphy',pad=20)
    ax =plt.subplot(1,2,2)
    ax=df_class['B'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
    ax.set_title(label = "Line B Fault", fontsize = 20,color='Brown',font='Lucida Calligraphy',pad=20);
    ax = plt.subplot(1,2,1)
    ax = sns.countplot(x='C', data=df_class)
    plt.title("Line C Fault", fontsize=20,color = 'Brown',font='Lucida Calligraphy',pad=20)
    ax =plt.subplot(1,2,2)
    ax=df_class['C'].value_counts().plot.pie(explode=[0.1, 0.1],autopct='%1.2f%%',shadow=True);
    ax.set_title(label = "Line C Fault", fontsize = 20,color='Brown',font='Lucida Calligraphy',pad=20);
    df_class['Fault_Type'] = df_class['G'].astype('str') + df_class['C'].astype('str') + df_class['B'].astype('str') + df_class['A'].astype('str')
    df_class.head().style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    df_class['Fault_Type'][df_class['Fault_Type'] == '0000' ] = 'NO Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '1001' ] = 'Line A to Ground Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '0110' ] = 'Line B to Line C Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '1011' ] = 'Line A Line B to Ground Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '0111' ] = 'Line A Line B Line C'
    df_class['Fault_Type'][df_class['Fault_Type'] == '1111' ] = 'Line A Line B Line C to Ground Fault'
    df_class.sample(10).style.set_properties(**{'background-color': 'blue',
                           'color': 'white',
                           'border-color': 'darkblack'})
    df_class.describe().style.background_gradient(cmap='rainbow')
    df_class['Fault_Type'].value_counts(ascending=False)
    ax = plt.figure(figsize = (15,16))
    ax = plt.subplot(2,1,1)
    ax = sns.countplot(x='Fault_Type', data=df_class)
    plt.title("Fault Type", fontsize=20,color = 'Brown',font='Lucida Calligraphy',pad=20)
    plt.xticks(rotation=65)
    plt.tight_layout()
    ax =plt.subplot(2,1,2)
    ax=df_class['Fault_Type'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1, 0.1,0.1],autopct='%1.2f%%',shadow=True);
    plt.tight_layout()
    plt.axis('off');
    plt.figure(figsize = (10,4))
    plt.plot(df_class["Ia"])
    plt.plot(df_class["Ib"])
    plt.plot(df_class["Ic"]);
    plt.figure(figsize = (10,4))
    plt.plot(df_class["Va"])
    plt.plot(df_class["Vb"])
    plt.plot(df_class["Vc"]);
    plt.figure(figsize= (15,10))
    plt.subplot(3,3,1)
    sns.distplot(df_class['Va'], rug = True, kde = False)
    plt.xlabel('Voltage in Per Unit(pu)', fontsize = 12)
    plt.title('Distribution of Voltage',fontsize = 15)
    plt.subplot(3,3,2)
    sns.distplot(df_class['Ia'], color= 'green',rug = True, kde = False)
    plt.title('Distribution of Load of Line',fontsize = 15)
    plt.xlabel('Load on line in Amperes', fontsize = 12)
    #Kde Plots
    plt.subplot(3,3,4)
    sns.kdeplot(df_class['Va'], shade = True)
    plt.xlabel('Voltage in Per Unit(pu)', fontsize = 12)
    plt.title('Distribution of Voltage',fontsize = 15)
    plt.subplot(3,3,5)
    sns.kdeplot(df_class['Ia'], shade = True, color = 'g')
    plt.title('Distribution of Load of Line',fontsize = 15)
    plt.xlabel('Load on line in Amperes', fontsize = 12)
    #Box Plots
    plt.subplot(3,3,7)
    sns.boxplot(x = df_class['Va'], orient = 'v',color= 'b', boxprops=dict(alpha=.5))
    plt.subplot(3,3,8)
    sns.boxplot(x = df_class['Ia'], orient = 'v', color= 'g', boxprops=dict(alpha=.5))
    plt.tight_layout()
    # plt.show()
    No_Fault = df_class[df_class['Fault_Type'] == 'NO Fault' ]
    No_Fault.sample(5).style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(No_Fault["Ia"],'r')
    ax = plt.plot(No_Fault["Ib"],'b')
    ax = plt.plot(No_Fault["Ic"],'g');
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(No_Fault["Va"],'r')
    ax = plt.plot(No_Fault["Vb"],'b')
    ax = plt.plot(No_Fault["Vc"],'g');
    plt.figure(figsize= (15,10))
    #plt.suptitle("Distributions of Different Features", fontsize = 20)
    #Histograms
    plt.subplot(3,3,1)
    sns.distplot(No_Fault['Va'], rug = True, kde = False)
    plt.xlabel('Voltage in Per Unit(pu)', fontsize = 12)
    plt.title('Distribution of Voltage',fontsize = 15)
    plt.subplot(3,3,2)
    sns.distplot(No_Fault['Ia'], color= 'green',rug = True, kde = False)
    plt.title('Distribution of Load of Line',fontsize = 15)
    plt.xlabel('Load on line in Amperes', fontsize = 12)
    #Kde Plots
    plt.subplot(3,3,4)
    sns.kdeplot(No_Fault['Va'], shade = True)
    plt.xlabel('Voltage in Per Unit(pu)', fontsize = 12)
    plt.title('Distribution of Voltage',fontsize = 15)
    plt.subplot(3,3,5)
    sns.kdeplot(No_Fault['Ia'], shade = True, color = 'g')
    plt.title('Distribution of Load of Line',fontsize = 15)
    plt.xlabel('Load on line in Amperes', fontsize = 12)
    #Box Plots
    plt.subplot(3,3,7)
    sns.boxplot(x = No_Fault['Va'], orient = 'v',color= 'b', boxprops=dict(alpha=.5))
    plt.subplot(3,3,8)
    sns.boxplot(x = No_Fault['Ia'], orient = 'v', color= 'g', boxprops=dict(alpha=.5))
    plt.tight_layout()
    # plt.show()
    Line_AG_Fault = df_class[df_class['Fault_Type'] == 'Line A to Ground Fault' ]
    Line_AG_Fault.head().style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_AG_Fault["Ia"],'r')
    ax = plt.plot(Line_AG_Fault["Ib"],'b')
    ax = plt.plot(Line_AG_Fault["Ic"],'g');
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_AG_Fault["Va"],'r')
    ax = plt.plot(Line_AG_Fault["Vb"],'b')
    ax = plt.plot(Line_AG_Fault["Vc"],'g');
    Line_ABG_Fault = df_class[df_class['Fault_Type'] == 'Line A Line B to Ground Fault' ]
    Line_ABG_Fault.head().style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_ABG_Fault["Ia"],'r')
    ax = plt.plot(Line_ABG_Fault["Ib"],'b')
    ax = plt.plot(Line_ABG_Fault["Ic"],'g');
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_ABG_Fault["Va"],'r')
    ax = plt.plot(Line_ABG_Fault["Vb"],'b')
    ax = plt.plot(Line_ABG_Fault["Vc"],'g');
    Line_BC_Fault = df_class[df_class['Fault_Type'] == 'Line B to Line C Fault' ]
    Line_BC_Fault.head().style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_BC_Fault["Ia"],'r')
    ax = plt.plot(Line_BC_Fault["Ib"],'b')
    ax = plt.plot(Line_BC_Fault["Ic"],'g');
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_BC_Fault["Va"],'r')
    ax = plt.plot(Line_BC_Fault["Vb"],'b')
    ax = plt.plot(Line_BC_Fault["Vc"],'g');
    Line_ABC_Fault = df_class[df_class['Fault_Type'] == 'Line A Line B Line C' ]
    Line_ABC_Fault.head().style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_ABC_Fault["Ia"],'r')
    ax = plt.plot(Line_ABC_Fault["Ib"],'b')
    ax = plt.plot(Line_ABC_Fault["Ic"],'g');
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_ABC_Fault["Va"],'r')
    ax = plt.plot(Line_ABC_Fault["Vb"],'b')
    ax = plt.plot(Line_ABC_Fault["Vc"],'g');
    Line_ABCG_Fault = df_class[df_class['Fault_Type'] == 'Line A Line B Line C to Ground Fault' ]
    Line_ABCG_Fault.head().style.set_properties(**{'background-color': 'blue',
                               'color': 'white',
                               'border-color': 'darkblack'})
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_ABCG_Fault["Ia"],'r')
    ax = plt.plot(Line_ABCG_Fault["Ib"],'b')
    ax = plt.plot(Line_ABCG_Fault["Ic"],'g');
    ax = plt.figure(figsize = (18,3))
    ax = plt.plot(Line_ABCG_Fault["Va"],'r')
    ax = plt.plot(Line_ABCG_Fault["Vb"],'b')
    ax = plt.plot(Line_ABCG_Fault["Vc"],'g');
    plt.figure(figsize= (15,10))
    #plt.suptitle("Distributions of Different Features", fontsize = 20)
    #Histograms
    plt.subplot(3,3,1)
    sns.distplot(Line_ABCG_Fault['Va'], rug = True, kde = False)
    plt.xlabel('Voltage in Per Unit(pu)', fontsize = 12)
    plt.title('Distribution of Voltage',fontsize = 15)
    plt.subplot(3,3,2)
    sns.distplot(Line_ABCG_Fault['Ia'], color= 'green',rug = True, kde = False)
    plt.title('Distribution of Load of Line',fontsize = 15)
    plt.xlabel('Load on line in Amperes', fontsize = 12)
    #Kde Plots
    plt.subplot(3,3,4)
    sns.kdeplot(Line_ABCG_Fault['Va'], shade = True)
    plt.xlabel('Voltage in Per Unit(pu)', fontsize = 12)
    plt.title('Distribution of Voltage',fontsize = 15)
    plt.subplot(3,3,5)
    sns.kdeplot(Line_ABCG_Fault['Ia'], shade = True, color = 'g')
    plt.title('Distribution of Load of Line',fontsize = 15)
    plt.xlabel('Load on line in Amperes', fontsize = 12)
    #Box Plots
    plt.subplot(3,3,7)
    sns.boxplot(x = Line_ABCG_Fault['Va'], orient = 'v',color= 'b', boxprops=dict(alpha=.5))
    plt.subplot(3,3,8)
    sns.boxplot(x = Line_ABCG_Fault['Ia'], orient = 'v', color= 'g', boxprops=dict(alpha=.5))
    plt.tight_layout()
    # plt.show()
    encoder = LabelEncoder()
    df_class['Fault_Type'] = encoder.fit_transform(df_class['Fault_Type'])
    df_class.head()
    X = df_class.drop(['Fault_Type','G','C','B','A'],axis=1)
    y = df_class['Fault_Type']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=21)
    X_train
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)

    log_train = round(logreg.score(X_train, y_train) * 100, 2)
    log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)

    print("Training Accuracy    :",log_train ,"%")
    print("Model Accuracy Score :",log_accuracy ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")
    print("Classification_Report: \n",classification_report(y_test,y_pred_lr))
    print("\033[1m--------------------------------------------------------\033[0m")
    cm = confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = logreg.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    # plt.show()
    decision = DecisionTreeClassifier()
    decision.fit(X_train, y_train)
    y_pred_dec = decision.predict(X_test)

    decision_train = round(decision.score(X_train, y_train) * 100, 2)
    decision_accuracy = round(accuracy_score(y_pred_dec, y_test) * 100, 2)

    print("Training Accuracy    :",decision_train ,"%")
    print("Model Accuracy Score :",decision_accuracy ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")
    print("Classification_Report: \n",classification_report(y_test,y_pred_dec))
    print("\033[1m--------------------------------------------------------\033[0m")
    cm = confusion_matrix(y_test, y_pred_dec, labels=decision.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = decision.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    # plt.show()
    def tree_plot(model_name):
        plt.figure(figsize=(20,20))
        features = df_class.columns
        classes = ['NO Fault','Line A to Ground Fault','Line B to Line C Fault','Line A Line B to Ground Fault','Line A Line B Line C','Line A Line B Line C to Ground Fault']
        tree.plot_tree(model_name,feature_names=features,class_names=classes,filled=True)
        # plt.show()
    prediction2 = decision.predict(X_test)
    print(prediction2)
    cross_checking = pd.DataFrame({'Actual' : y_test , 'Predicted' : prediction2})
    cross_checking.sample(5).style.background_gradient(
            cmap='coolwarm').set_properties(**{
                'font-family': 'Lucida Calligraphy',
                'color': 'LigntGreen',
                'font-size': '15px'
            })
    X_test
    y
    ans=decision.predict(X_test)
    ans
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)

    log_train = round(logreg.score(X_train, y_train) * 100, 2)
    log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)


    print("Training Accuracy    :",log_train ,"%")
    print("Model Accuracy Score :",log_accuracy ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")
    print("Classification_Report: \n",classification_report(y_test,y_pred_lr))
    print("\033[1m--------------------------------------------------------\033[0m")
    cm = confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = logreg.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    # plt.show()
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    random_forest.score(X_train, y_train)

    random_forest_train = round(random_forest.score(X_train, y_train) * 100, 2)
    random_forest_accuracy = round(accuracy_score(y_pred_rf, y_test) * 100, 2)

    print("Training Accuracy    :",random_forest_train ,"%")
    print("Model Accuracy Score :",random_forest_accuracy ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")
    print("Classification_Report: \n",classification_report(y_test,y_pred_rf))
    print("\033[1m--------------------------------------------------------\033[0m")
    cm = confusion_matrix(y_test, y_pred_rf, labels=random_forest.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = random_forest.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    # plt.show()
    xgb = Xgb.XGBClassifier()
    xgb.fit(X_train,y_train)
    y_pred_xgb = xgb.predict(X_test)
    xgb.score(X_train, y_train)

    xgb_train = round(xgb.score(X_train, y_train) * 100, 2)
    xgb_accuracy = round(accuracy_score(y_pred_xgb, y_test) * 100, 2)

    print("Training Accuracy    :",xgb_train ,"%")
    print("Model Accuracy Score :",xgb_accuracy ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")
    print("Classification_Report: \n",classification_report(y_test,y_pred_xgb))
    print("\033[1m--------------------------------------------------------\033[0m")
    cm = confusion_matrix(y_test, y_pred_xgb, labels=xgb.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = xgb.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    # plt.show()
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)

    svc_train = round(svc.score(X_train, y_train) * 100, 2)
    svc_accuracy = round(accuracy_score(y_pred_svc, y_test) * 100, 2)

    print("Training Accuracy    :",svc_train ,"%")
    print("Model Accuracy Score :",svc_accuracy ,"%")
    print("\033[1m--------------------------------------------------------\033[0m")
    print("Classification_Report: \n",classification_report(y_test,y_pred_svc))
    print("\033[1m--------------------------------------------------------\033[0m")
    cm = confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = logreg.classes_)
    disp.plot()
    plt.title('Confusion Matrix')
    # plt.show()
    models = pd.DataFrame({
        'Model': [
            'Support Vector Machines', 'Logistic Regression', 'Random Forest',
            'Decision Tree', 'XGBClassifier'
        ],

        'Training Accuracy':
        [log_train, svc_train, decision_train, random_forest_train, xgb_train],

        'Model Accuracy Score': [
            log_accuracy, svc_accuracy, decision_accuracy, random_forest_accuracy,
            xgb_accuracy
        ]
    })
    pd.set_option('display.precision',2)
    models.sort_values(by='Model Accuracy Score', ascending=False).style.background_gradient(
            cmap='coolwarm').set_properties(**{
                'font-family': 'Lucida Calligraphy',
                'color': 'LigntGreen',
                'font-size': '15px'
            })
    x={"Ia":[ia],"Ib":[ib],"Ic":[ic],"Va":[va],"Vb":[vb],"Vc":[vc]}
    y=pd.DataFrame(x)
    y
    ans=decision.predict(y)
    mapping_dict = {
        0 : "LLL Fault (Between Phases A, B and C)",
        1 : "LLLG Fault (Three Phase Symmetrical Fault)",
        2 : "LLG Fault (Between Phases A, B and Ground)",
        3 : "LG Fault (Between Phases A and Ground)",
        4 : "LL Fault (Between Phase B and Phase C)",
        5 : "No Fault"
    }
    if ans[0] in mapping_dict:
        return mapping_dict[ans[0]]
    return ans[0]
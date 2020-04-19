import pandas as pd
import numpy as np
import os, re, json
from datetime import date 
from keras import utils
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import shutil

from string import punctuation
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow_hub as hub
from rule_engine import RuleIntel

def alarm_pred(dataframe):
    df=pd.read_excel(dataframe)

    #drop unused features
    df_feat=df.drop(['LASTFAULTSUMMARY','FIRSTOCCURRENCE','LASTOCCURRENCE','MASTER','MASTERSERVICE'],axis=1)

    #encode feeatures
    df_feat['TROUBLETICKET']=df_feat['TROUBLETICKET'].apply(lambda x:1 if type(x)==str else 0)
    df_feat['AlertType']=df_feat['Alert type'].apply(lambda x: 1 if (x)==True else 0)


    df_feat_X = df_feat[['NODE','ALARMKEY','FIRSTEVENTSUMMARY','HIGHESTSEVERITY','CLASS','AlertType']] 

    #replace missing values
    df_feat_X['ALARMKEY']=df_feat_X['ALARMKEY'].apply(lambda x:x if type(x)==str else 'other')

    #drop na
    df_feat_X.dropna(inplace=True)

    #encoding for features

    le = preprocessing.LabelEncoder()
    df_feat_X['NODE_CAT_MAP'] = le.fit_transform(df_feat_X['NODE'])
    le = preprocessing.LabelEncoder()
    df_feat_X['ALARMKEY_CAT_MAP']=le.fit_transform(df_feat_X['ALARMKEY'])
    le = preprocessing.LabelEncoder()
    df_feat_X['HIGHESTSEVERITY_CAT_MAP']=le.fit_transform(df_feat_X['HIGHESTSEVERITY'])
    le = preprocessing.LabelEncoder()
    df_feat_X['CLASS_CAT_MAP']=le.fit_transform(df_feat_X['CLASS'])

    #Converted features
    df_feat_conv = df_feat_X[['FIRSTEVENTSUMMARY','NODE_CAT_MAP','ALARMKEY_CAT_MAP','HIGHESTSEVERITY_CAT_MAP','CLASS_CAT_MAP']]

    #taking X and y
    X = df_feat_conv
    y= df_feat_X['AlertType']

    #Scaling
    scaler = StandardScaler()

    #meta train test split
    # X_train_meta, X_test_meta, y_train, y_test = train_test_split((X.drop('FIRSTEVENTSUMMARY',axis=1)), (y) ,stratify=y, test_size=0.20,random_state=10)


    #scale X
    standardized = scaler.fit(X.drop('FIRSTEVENTSUMMARY',axis=1).values)
    X__std = standardized.transform(X.drop('FIRSTEVENTSUMMARY',axis=1).values)
    # X_test_std = standardized.transform(X.values)

    #Getting the transfer learning embeddings
    embedding_128 = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

    #creating the embedding confuguration 
    hub_layer = hub.KerasLayer(embedding_128, input_shape=[], 
                            dtype=tf.string, trainable=True)

    #NLP feature
    X_summary = df_feat_conv[['FIRSTEVENTSUMMARY']]
    X_nlp = X_summary
    # X_train_nlp, X_test_nlp, y_train, y_test = train_test_split(X_summary, (y) ,stratify=y, test_size=0.20,random_state=10)
    X_nlp['FIRSTEVENTSUMMARY'] = X_nlp['FIRSTEVENTSUMMARY'].apply(lambda x: x.lower())
    # X_train_nlp['FIRSTEVENTSUMMARY'] = X_train_nlp['FIRSTEVENTSUMMARY'].apply(lambda x: x.lower())
    # X_test_nlp['FIRSTEVENTSUMMARY'] = X_test_nlp['FIRSTEVENTSUMMARY'].apply(lambda x: x.lower())

    #NPL Preprocessing 
    X_nlp['FIRSTEVENTSUMMARY'] = X_nlp['FIRSTEVENTSUMMARY'].apply(lambda s: re.sub(r"[^a-zA-Z0-9]"," ",s))

    # X_train_nlp['FIRSTEVENTSUMMARY'] = X_train_nlp['FIRSTEVENTSUMMARY'].apply(lambda s: re.sub(r"[^a-zA-Z0-9]"," ",s))
    # X_test_nlp['FIRSTEVENTSUMMARY'] = X_test_nlp['FIRSTEVENTSUMMARY'].apply(lambda s: re.sub(r"[^a-zA-Z0-9]"," ",s))


    lemmatizer=WordNetLemmatizer()

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        tokens =[w for w in tokens if not w in stop_words] # [w for w in
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
        return lemmatized_output
    #apply tokenize and lemmatize function
    X_nlp['FIRSTEVENTSUMMARY']=X_nlp['FIRSTEVENTSUMMARY'].apply(tokenize)

    # X_train_nlp['FIRSTEVENTSUMMARY']=X_train_nlp['FIRSTEVENTSUMMARY'].apply(tokenize)
    # X_test_nlp['FIRSTEVENTSUMMARY']=X_test_nlp['FIRSTEVENTSUMMARY'].apply(tokenize)

    ### Data in embed 
    embed = (hub_layer(X_nlp.values.reshape(-1,)))
    embed_df = pd.DataFrame((embed.numpy()[:].reshape(-1,128)))
    X_std_df = pd.DataFrame(X__std)
    concat_df = pd.concat([X_std_df,embed_df] ,axis=1)

    # loding model
    model = tf.keras.models.load_model("MABinary_model.h5")

    y_pred = model.predict_on_batch(concat_df.values)

    output = pd.DataFrame(columns=['y_pred'],data = y_pred.numpy())
    output['pred_val'] = output['y_pred'].apply(lambda x:1 if (x) >= .58 else 0)

    #Generate false alarm excel
    false_output = output[output['pred_val']==0]
    print(false_output.shape)
    output_df_false = df[df.index.isin(false_output.index)]
    output_df_false.to_excel('false_alarm_report.xls')


    #Generating true alarms from predicted output
    true_output = output[output['pred_val']==1]
    print(true_output.shape)
    output_df_true =df[df.index.isin(true_output.index)]
    print(output_df_true)
    print(output_df_true.shape)

    #load dict json contained mapped rules
    rulejson = json.load(open('dict_map_rule.json'))
    train = pd.read_excel('dummy.xls')

    train['FIRSTOCCURRENCE'] = pd.to_datetime(train['FIRSTOCCURRENCE'], format=None,
        exact=True,
        infer_datetime_format=True,dayfirst=True)

    output_df_true['FIRSTOCCURRENCE'] = pd.to_datetime(output_df_true['FIRSTOCCURRENCE'], format=None,
        exact=True,
        infer_datetime_format=True,dayfirst=True)

    x=1

    for i,value in train.sort_values('FIRSTOCCURRENCE').iterrows():
        #print(i,value[2])
        train.loc[i,'Count']= x
        #df_test.loc[i,'BMC_Status'] = 'Open'
        train.loc[i,'BMC_Status'] = 'Closed'
        train.loc[i,'Rule_Action'] = 'action obsolete'
        train.loc[i,'Rule_Ticket'] = 'ticket obsolete'
        #df.loc[i,'timediff']= (df[df.Val == 'ticket3'].loc[i,'Date'] - df[df.Val == 'ticket3'].loc[i-1,'Date']).total_seconds() / 60
        x+=1
    x=0

    x=1
    for i,value in output_df_true.sort_values('FIRSTOCCURRENCE').iterrows():
        #print(i,value[2])
        output_df_true.loc[i,'Count']= ''
        #df_test.loc[i,'BMC_Status'] = 'Open'
        output_df_true.loc[i,'BMC_Status'] = ''
        output_df_true.loc[i,'Rule_Action'] = ''
        output_df_true.loc[i,'Rule_Ticket'] = ''
        #df.loc[i,'timediff']= (df[df.Val == 'ticket3'].loc[i,'Date'] - df[df.Val == 'ticket3'].loc[i-1,'Date']).total_seconds() / 60
        x+=1
    x=0

    #current time
    rng = pd.date_range(date.today(), periods=35000, freq='T')
    #read combine rule book
    df_cat3 = pd.read_excel(r"combine_rulebook.xls")

    h=0
    df_in_alarm=''
    for i,value in output_df_true.sort_values('FIRSTOCCURRENCE').iterrows():
    #     print(h)
        df_in_alarm = output_df_true.loc[[i]].to_dict('r')##dict
        df_in_alarm[0]['FIRSTOCCURRENCE'] = rng[h]
        df_in_alarm[0]['BMC_Status'] = 'Open'
        df_in_alarm[0]['Rule_Action'] = ''
        df_in_alarm_frame = pd.DataFrame(df_in_alarm)
        innit_AlarmDateCount = RuleIntel()
        innit_AlarmDateCount.load_input(df_in_alarm[0],train)
        innit_AlarmDateCount.getlastalarmcount()
        train = innit_AlarmDateCount.appendrecord()
        innit_AlarmDateCount.gettimeinterval(train)
        train = innit_AlarmDateCount.Map_RuleId(df_in_alarm_frame,train,df_cat3,rulejson)
        h+=1

    #Generating alarm action report
    report = train.drop(['Rule_Ticket'],axis=1)
    print(report)
    report.to_excel("alarm_action_report.xls")

    ##Duplication Function##
    
    dupdf = pd.DataFrame(columns = ['NODE','CLASS', 'FIRSTEVENTSUMMARY','Count', 'BMC_Status', 'Rule_Action',
       'Rule_Ticket','Duplicate'])
    
    for l,valu in train[train['Rule_Ticket'] == 1][['NODE', 'CLASS', 'FIRSTEVENTSUMMARY', 'Count', 'BMC_Status',
       'Rule_Action', 'Rule_Ticket']].iterrows():
        if (len(dupdf[(dupdf['NODE']== valu['NODE']) & (dupdf['CLASS']== valu['CLASS']) & 
                        (dupdf['FIRSTEVENTSUMMARY']== valu['FIRSTEVENTSUMMARY']) & 
                        (dupdf['BMC_Status']=='Open')]) >=1):
            valu['Duplicate'] = 'Duplicate'
            dupdf = dupdf.append(valu,ignore_index=True)
        else:
            valu['Duplicate'] = 'Create Ticket'
            dupdf = dupdf.append(valu,ignore_index=True)
        
    dupdf.to_excel('duplication_alarm_report.xls')

 
    return 'sucess'







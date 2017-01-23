import kagglegym
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from collections import Counter
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import geometric_mean_score



env = kagglegym.make()
o = env.reset()
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
feature_cols = [c for c in o.train.columns if c not in excl] ##original feature cols

df_train = o.train
df_test = o.features
df_test['y'] = o.target



def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def preprocessing(df):
    # input parameter: o.train
    
    ## drop cols with over 30% missing value
    missing_ratio = df.isnull().sum()/df.shape[0]
    col_to_drop = missing_ratio.where(lambda x:x>0.3).dropna().index
    df = df.drop(col_to_drop, axis=1)
    
    
    ## calculate the median for each column before filling missing value
    d_median= df.median(axis=0)
    n = df.isnull().sum(axis=1)
    
    for c in feature_cols:
        df[c + '_nan_'] = pd.isnull(df[c]) ## add col to indicate if the number is null or not
        d_median[c + '_nan_'] = 0

    ## forward fill the missing value, gap limit set to be three
    df_ffilled = df.set_index(['id','timestamp']).sort_index().fillna(method = 'ffill',limit =3).reset_index()
    
    ## fill the other missing values with median
    df_filled = df_ffilled.fillna(d_median)
    
    print('filled...')
    
    ## add another col to indicate number of missing value
    df_filled['znull'] = n
    
    print('missing value captured...')
    
    ## transform values into their reciprocals for more centralized distribution
    
    df_transformed = df_filled.copy()
    
    non_transformable_fea = []
    
    for feature in feature_cols:
        try:
            transformed_list = list(map(lambda x:1/x,df_filled[feature]))
            df_transformed[feature]=transformed_list 
        except OverflowError:
            non_transformable_fea.append(feature)
        except ValueError:
            non_transformable_fea.append(feature)
    
    ## add col to indicate if the value is outlier or not
    
    for feature in feature_cols:
        
        df_transformed[feature+'outlier'] = mad_based_outlier(df_filled[feature])
        
    print('transformed...')
    
    ## print the dimension of the input features
    
    print('till now , we have %d features'%(len(df_transformed.columns)-1))
    
    return df_transformed


def shuffling(df):
    df_shuffled = shuffle(df,random_state=0).reset_index(drop = True)
    
    return df_shuffled

## use isolation forest to detect anomaly
def anomaly_isolation(df):
    
    # parameter: transformed df
    
    df_shuffled = shuffling(df)
    X = df_shuffled[feature_cols]
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=100, random_state=rng,contamination=0.05)
    clf.fit(X)
    anomaly_pred = clf.predict(X)
    df_shuffled['ano_y_iso'] = anomaly_pred
    return df_shuffled
    
    
### classification

def classifier_model(df):
    # parameter:   original train data
    df_transformed = preprocessing(df)
    df_shuffled = anomaly_isolation(df_transformed)
    
    y  = mad_based_outlier(df_shuffled['y']) #label
    X = df_shuffled[feature_cols] ##use original features to do classification
    
    
    RANDOM_STATE = 42
    pipeline = pl.make_pipeline(os.SMOTE(random_state=RANDOM_STATE),
                            RandomForestClassifier(random_state=RANDOM_STATE))
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=RANDOM_STATE)
    
    # Train the classifier with balancing
    pipeline.fit(X_train, y_train)    
    print('The geometric mean is {}'.format(geometric_mean_score(
    y_test,
    y_pred_bal)))
    
    return pipeline

model_anomaly_pipepline = classifier_model(df_train)


def anomaly_rf(df):
    ## shuffled df
    
    feature_cols_all = [c for c in df.columns if c not in excl] ##original feature cols

    df['anomaly_classifier'] = model_anomaly_pipepline.predict(df[feature_cols_all])
     
    return df



def feature_engineering(df):
    
    df_transformed = preprocessing(df)
    df_shuffled = anomaly_isolation(df_transformed)
    df_final = anomaly_rf(df_shuffled)
    
    return df_final

## feature engineering for train and test:
df_train_processed = feature_engineering(df_train)
df_test_processed = feature_engineering(df_test_X)



# Modeling

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, o.train['y'])

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), o.train.loc[y_is_within_cut, 'y'])
train = []

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())

while True:
    test = o.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    pred = o.target
    test2 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.65) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * 0.35)
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
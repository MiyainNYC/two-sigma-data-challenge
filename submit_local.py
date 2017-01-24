import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split


with pd.HDFStore("data/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")

excl = ['id', 'timestamp', 'y']  ##original feature cols
df_train = df.loc[:855377]
df_test = df.loc[855378:]

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
    print('col with large missing value removed...')

    feature_cols = [x for x in list(df.columns) if x not in excl]
    print(feature_cols)
    
    ## calculate the median for each column before filling missing value
    d_median = df.median(axis=0)
    n = df.isnull().sum(axis=1)
    
    for c in feature_cols:
        df[c + '_nan'] = pd.isnull(df[c]) ## add col to indicate if the number is null or not
        d_median[c + '_nan'] = 0
        print('%s nan catched..'%c)

    print('creat col indicating nan...')

    ## forward fill the missing value, gap limit set to be three
    df_ffilled = df.set_index(['id','timestamp']).sort_index().fillna(method = 'ffill',limit =3).reset_index()
    print('forward fill completed...')
    ## fill the other missing values with median
    df_filled = df_ffilled.fillna(d_median)
    ## add another col to indicate number of missing values
    df_filled['znull'] = n

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
    print('transformed...')
    ## add col to indicate if the value is outlier or not
    
    for feature in feature_cols:
        
        df_transformed[feature+'outlier'] = mad_based_outlier(df_filled[feature])
    print('add outliers detection cols with MAD...')

    
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
    print('data shuffled for isolation forest...')
    feature_cols = [x for x in list(df.columns) if x not in excl]
    X = df_shuffled[feature_cols]
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=100, random_state=rng,contamination=0.05)
    clf.fit(X)
    anomaly_pred = clf.predict(X)
    df_shuffled['ano_y_iso'] = anomaly_pred
    print('add outliers detection cols with isolation forest(unsupervised)...')

    return df_shuffled
    
    
### classification

def classifier_model(df):
    # parameter:   original train data
    df_transformed = preprocessing(df)
    df_shuffled = anomaly_isolation(df_transformed)

    feature_cols = [c for c in df_shuffled.columns if c not in excl]

    y = mad_based_outlier(df_shuffled['y'])  # label
    X = df_shuffled[feature_cols]  ##use original features to do classification

    RANDOM_STATE = 42
    pipeline = pl.make_pipeline(os.SMOTE(random_state=RANDOM_STATE),
                                RandomForestClassifier(random_state=RANDOM_STATE))
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=RANDOM_STATE)

    # Train the classifier with balancing
    pipeline.fit(X_train, y_train)
    y_pred_bal = pipeline.predict(X_test)

    print('for supervised outlier detection, the geometric mean is {}'.format(geometric_mean_score(
        y_test,
        y_pred_bal)))

    return pipeline


model_anomaly_pipepline = classifier_model(df_train)
print('Random forest classifier model prepared...')


def anomaly_rf(df):
    ## shuffled df

    feature_cols_all = [c for c in df.columns if c not in excl]  ##original feature cols

    df['anomaly_classifier'] = model_anomaly_pipepline.predict(df[feature_cols_all])

    print('add outlier detection cols with random forest(supervised)...')


    return df


##############################################
#############################################


#######################################################

def feature_engineering(df):
    
    df_transformed = preprocessing(df)
    df_shuffled = anomaly_isolation(df_transformed)
    #f_final = anomaly_rf(df_shuffled)
    df_final = df_shuffled
    
    return df_final

## feature engineering for train and test:
print('PROCESS TRAIN DATA')
df_train_processed = feature_engineering(df_train)
print('----------------------------------------------')
print('----------------------------------------------')

print('PROCESS TEST DATA')
df_test_processed = feature_engineering(df_test)

features = [c for c in df_train_processed.columns if c not in excl]

# Modeling

print('MODELING')
rfr = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model = rfr.fit(df_train_processed[features],df_train_processed['y'])

print(model.predict(df_test_processed.sort_values(by = 'id')))

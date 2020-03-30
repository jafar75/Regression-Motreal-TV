import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump, load


def apply_hashing(feature, feature_len):
    column_names = []
    for i in range(feature_len): column_names.append(feature.name + "_" + str(i))

    h = FeatureHasher(n_features=feature_len, input_type="string")
    f = h.transform(feature)
    return pd.DataFrame(f.toarray(), columns=column_names)


def clean_data(df):
    # dealing with column names
    df.rename(columns={"Temperature in Montreal during episode": "Temperature",
                       "Game of the Canadiens during episode?": "Game_of_canadiains",
                       "# of episode in the season": "Multiple_episode",
                       "Name of episode": "Name_episode",
                       "Day of week": "Weekday",
                       "Channel Type": "Channel_type",
                       "First time or rerun": "First_time",
                       "Market Share_total": "Total_market_share",
                       "Movie?": "Movie"}, inplace=True)

    # removing first column (unnamed: 0) which is not necessary
    df.drop(columns=['Unnamed: 0'], inplace=True)
    # checking columns of `Episod` and `Name of show` values
    equal_two_columns = df['Episode'].equals(df['Name of show'])

    # Actually these 2 columns are exactly the same
    if equal_two_columns:  # True
        df.drop(columns=['Name of show'], inplace=True)

    # handling "Name of episode" column
    different_categories = len(
        df['Episode'].unique())  # different categories is very large. not possible to use one-hot encoding.

    # filling null values
    df['Name_episode'] = df.apply(
        lambda row: row['Episode'] if pd.isnull(row['Name_episode']) else row['Name_episode'],
        axis=1
    )

    # Handling null values in Start/End_time
    # Very few records have null values in `Start_time` and `End_time`. we remove them.
    df = df.dropna(subset=['Start_time', 'End_time'])

    # Handling and cleanning temperature column.
    # becuase rows are sorted with Start_time in one station we can use interpolation to fill nulls.
    df['Temperature'].interpolate(inplace=True)
    df.Temperature = df.Temperature.astype(int)

    # handling date and time features
    # By analysing the dataset it was realized that `length` column show each 15 minutes of the episode.
    # We can compute exact length of episode with `Start_time` and `End_time` as minute
    # So `End_time` is redundant maybe and it can be removed

    # we could simply multiply `length` to 15. But some episodes that have length less than 15 are set to 0 in `Length` column.

    df['Start_time'] = pd.to_timedelta(pd.to_datetime(df['Start_time']))
    df['End_time'] = pd.to_timedelta(pd.to_datetime(df['End_time']))

    df['Length'] = df.apply(
        lambda row: (row.End_time.total_seconds() - row.Start_time.total_seconds()) / 60,
        axis=1
    )
    df.Length = df.Length.astype(int)

    df.drop(['End_time'], inplace=True, axis=1)

    # Also we should extract information from date and time features as categorical features
    df['Date'] = pd.to_datetime(df.Date)
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['Start_time'] = pd.to_datetime(df['Start_time'])
    df['Hour'] = df.Start_time.dt.hour
    df['Minute'] = df.Start_time.dt.minute

    df.drop(['Date', 'Start_time'], inplace=True, axis=1)

    # Checking for rows with invalid length (negative)
    df = df[df.Length >= 0]

    # Some convertions. some of the features have Yes/No format or have few different categories.
    df.First_time.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    df.Multiple_episode.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    df.Movie.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    df.Game_of_canadiains.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)
    df.Channel_type.replace(to_replace=['General Channel', 'Specialty Channel'], value=[0, 1], inplace=True)
    df.Season.replace(to_replace=['Spring', 'Summer', 'Fall', 'Winter'], value=[0, 1, 2, 3], inplace=True)

    df.reset_index(drop=True, inplace=True)

    hashed_episode = apply_hashing(df.Episode, 10)
    hashed_name_episode = apply_hashing(df.Name_episode, 8)
    hashed_station = apply_hashing(df.Station, 4)
    hashed_genre = apply_hashing(df.Genre, 4)
    hashed_weekday = apply_hashing(df.Weekday, 3)  # Maybe better to use one-hot

    df = pd.concat([df, hashed_episode, hashed_name_episode, hashed_station, hashed_genre, hashed_weekday],
                   axis=1).drop(
        ['Name_episode', 'Episode', 'Station', 'Genre', 'Weekday'], axis=1)
    return df


def train_model(original_df):
    df = original_df.copy()
    y = df['Total_market_share']
    df.drop(columns=['Total_market_share'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=42)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    regr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                 max_depth=15, max_features=None, max_leaf_nodes=None,
                                 max_samples=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, min_samples_leaf=2,
                                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                                 n_estimators=300, n_jobs=3, oob_score=False,
                                 random_state=42, verbose=5, warm_start=False)
    regr.fit(X_train, y_train)

    y_train_prd = regr.predict(X_train)
    y_test_prd = regr.predict(X_test)

    r2_score_train = r2_score(y_train, y_train_prd)
    r2_score_test = r2_score(y_test, y_test_prd)

    print('r2_score Train: ', r2_score_train)
    print('r2_score Test: ', r2_score_test)
    print('...................................')

    mae_score_train = mean_absolute_error(y_train, y_train_prd)
    mae_score_test = mean_absolute_error(y_test, y_test_prd)

    print('mae_score Train: ', mae_score_train)
    print('mae_score Test: ', mae_score_test)

    return regr


def save_model(model):
    dump(model, '../models/final_model.joblib')


def load_model(path):
    return load(path)


##########
# Opening the csv file and some basic analysis
data_path = "../data/data.csv"
df = pd.read_csv(data_path)
df.info()

# Data cleaning
df = clean_data(df)
df.head()

# Training phase
final_model = train_model(df)

# Saving the Model
save_model(final_model)

# Prediction on the unseen data
test_data_path = "../data/test.csv"
test_df = pd.read_csv(test_data_path)
test_df = clean_data(test_df)

loaded_model = load_model('../models/final_model.joblib')
prd = loaded_model.predict(test_df)
df_prd = pd.DataFrame(prd, columns=['Market Share_total'])
df_prd.to_csv('../data/prediction.csv', index=False)

import pandas as pd
from sklearn.preprocessing import StandardScaler

#available metrics in csv
FEATURES = [
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "key_passes_per90",
    "tackles_per90",
    "interceptions_per90"
]

def preprocess_players(players: pd.DataFrame):
    '''
    Prepare player stats for machine leraning.
    Arguments | players: Dataframe containing players stats
    Returns | scaled_data: standardized numerical features
              scaler: fitted scaler for new data transform
    '''
    #select player metrics used by the model
    data = players[FEATURES].copy()
    #validate missing values
    if data.isna().any().any():
        raise ValueError("Player data contains missing values")
    #scale metrics - z = (x-mu)/sigma
    #standardscaler transforms feature to mean 0 and variance 1
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler
import numpy as np
import pandas as pd
from os import getcwd, listdir
from sqlalchemy import create_engine
from getpass import getpass

# Create Connection
user = 'postgres'
pw = getpass("Type in your password: ")
host = '104.225.218.162'
db = 'gender_prediction'

con = create_engine(f"postgresql://{user}:{pw}@{host}:8201/{db}").connect()

# Read in National Names dataset
data_path = getcwd() + "/Gender_Prediction/NationalNames.csv"
names_df = pd.read_csv(data_path, dtype={'Count': np.int32})

# Write Data to SQL Table in gender_prediction DB
names_df.to_sql("NationalNames", con=con, index=False)

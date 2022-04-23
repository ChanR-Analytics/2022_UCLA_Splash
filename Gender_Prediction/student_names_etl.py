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

# Get Student Names from HTML and Read in Pandas
data_path = getcwd() + "/Gender_Prediction/student_names_html.txt"

with open(data_path, 'r') as my_file:
    txt = my_file.read()

student_df = pd.read_html(txt)

student_df[0].iloc[1:].to_sql("M246_Student_Data", con=con, index=False)

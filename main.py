import os
import pandas as pd
import json

def create_data_dict(directory):
    data_dict = {}
    for filename in os.listdir(directory):
        company_name = filename.split(".")[0]
        company_dict = {}
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        df = df.dropna()
        df['pct_change'] = df["Adj Close"].pct_change()
        df.loc[0, "pct_change"] = 0
        for _, row in df.iterrows():
            company_dict[row["Date"]] = {
                "price": row["pct_change"]
            }
        data_dict[company_name] = company_dict
    return data_dict

directory = "data/price/raw"
data = create_data_dict(directory)

print(len(data.items()))
with open('data.txt', 'w') as data_file: 
     data_file.write(json.dumps(data))

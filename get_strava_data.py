import requests
import os
import glob
import pandas as pd
from dotenv import load_dotenv
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"

def get_access_token():
    payload = {
        'client_id': os.environ.get("client_id"),
        'client_secret': os.environ.get('client_secret'),
        'refresh_token': os.environ.get('refresh_token'),
        'grant_type': "refresh_token",
        'f': 'json'
    }

    print("Requesting Token...\n")
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    print("Access Token = {}\n".format(access_token))
    return access_token

def save_csv(dataframe, filename):
    print(f'Saving {filename}')
    dataframe.to_csv(filename)

def merge_files(path, filename):
    print('Merging files')
    csv_files = [pd.read_csv(_file)
                 for _file in glob.glob(os.path.join(path, "*.csv"))]
    final_df = csv_files.pop(len(csv_files)-1)
    final_df = final_df.append(csv_files)
    save_csv(final_df, filename)

def get_data(url, access_token, numb_items, page):
    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': numb_items, 'page': page}
    data = requests.get(url, headers=header, params=param).json()
    dataframe = pd.json_normalize(data)
    return dataframe

def main():
    access_token = get_access_token()
    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 1}
    data = requests.get(activites_url, headers=header, params=param).json()
    page = 1
    print("Retrieving Strava Data\n")
    while True:
        response = get_data(activites_url, access_token, 200, page)
        if 'message' in response.columns:
            raise Exception("Authorization Error: Check get_access_token()")
        if response.empty:
            break
        save_csv(response, f'data/strava_activities_page_{page}.csv')
        page += 1
    merge_files('data/', 'result/all_strava_activities.csv')
    print('Strava Data Retrieval Successful')

if __name__ == '__main__':
    main()
import subprocess
import calendar
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('result/all_strava_activities.csv')
print('Dataframe Shape: ', df.shape)

null_df = [[col, df[col].isnull().sum()] for col in df.columns]
print('Null Data:', df.isnull().sum().sum())
list(filter(lambda x: x[1]>0, null_df))

selected_columns = ['distance', 'moving_time', 'elapsed_time', 'sport_type', 'id', 'start_date_local',
                    'achievement_count', 'comment_count', 'athlete_count', 'start_latlng', 'end_latlng',
                    'average_speed', 'max_speed', 'average_heartrate', 'max_heartrate', 'elev_high', 'elev_low',
                    'upload_id', 'external_id', 'pr_count', 'map.summary_polyline']
df = df[selected_columns]

df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
df = df.sort_values(by='start_date_local')

df['weekday'] = df['start_date_local'].map(lambda x: x.weekday)
df['start_time'] = df['start_date_local'].dt.time
df['start_time'] = df['start_time'].astype(str)
df['start_date'] = df['start_date_local'].dt.date

df = df.drop('start_date_local', 1)

df['elev_high'] = df['elev_high'].fillna(value=0)
df['elev_low'] = df['elev_low'].fillna(value=0)
df['upload_id'] = df['upload_id'].fillna(value='unknown')
df['external_id'] = df['external_id'].fillna(value='unknown')
df['map.summary_polyline'] = df['map.summary_polyline'].fillna(value='unknown')
df['average_heartrate'] = df['average_heartrate'].fillna(value=df['average_heartrate'].mean())
df['max_heartrate'] = df['max_heartrate'].fillna(value=df['max_heartrate'].mean())

df['moving_time_minutes'] = round(df['moving_time']/60, 2)
df['distance_mi'] = df['distance'] / 1609.344
df['pace'] = df['moving_time_minutes'] / df['distance_mi']

df['elev'] = df['elev_high'] - df['elev_low']
df['year']= df['start_date'].map(lambda x: x.year)

runs = df.loc[df['sport_type'] == 'Run']

# def get_city_state_from_value(value):
#     if value == '[]':
#         return 'unknown'
#     value = value.replace('[','').replace(']','')
#     if value != ['']:
#         result = geolocator.reverse(value).address
#     else:
#         result = 'unknown'
#     return result

# geolocator = Nominatim(user_agent="strava_exploration_data")
# df['location'] = df['start_latlng'].map(get_city_state_from_value)

df['pace_sub_6'] = np.where(df['pace']<=6, True, False)

def workout_by_year():
    fig = sns.catplot(x='year', hue='sport_type', data=df, kind='count')
    fig.fig.suptitle('Workouts by Year')
    fig.set_xlabels('Year')
    fig.set_ylabels('Count')
    fig

def time_vs_elevation():
    p = sns.regplot(x='moving_time_minutes', y = 'elev', data=runs)

    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                        y=p.get_lines()[0].get_ydata())

    sns.regplot(x='moving_time_minutes', y = 'elev', data=runs).set_title("Distance vs Elevation")

    print('Slope: ' + str(slope))

def distance_vs_elevation():
    p = sns.regplot(x='distance', y = 'elev', data=runs)

    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                        y=p.get_lines()[0].get_ydata())

    sns.regplot(x='distance', y = 'elev', data=runs).set_title("Distance vs Elevation")

    print('Slope: ' + str(slope))

def average_time_per_day():
    runs.groupby('weekday').mean()['moving_time_minutes'].plot.bar()

def average_distance_per_day():
    runs.groupby('weekday').mean()['distance'].plot.bar()

def average_pace_vs_moving_time():
    p = sns.regplot(x='moving_time_minutes', y = 'pace', data=runs)

    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                        y=p.get_lines()[0].get_ydata())

    sns.regplot(x='moving_time_minutes', y = 'pace', data=runs).set_title("Average Pace vs Moving Time")

    print('Slope: ' + str(slope))

def average_pace_vs_distance():
    p = sns.regplot(x='distance', y = 'pace', data=runs)

    slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                        y=p.get_lines()[0].get_ydata())

    sns.regplot(x='distance', y = 'pace', data=runs).set_title("Average Pace vs Distance")

    print('Slope: ' + str(slope))

def average_pace_over_time():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x = np.asarray(runs.start_date)
    y = np.asarray(runs.pace)

    ax1.plot_date(x, y)
    ax1.set_title('Average Pace Over Time')

    x2 = mdates.date2num(x)
    z=np.polyfit(x2,y,1)
    p=np.poly1d(z)
    plt.plot(x,p(x2),'r--')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.show()
    print('Slope: ' + str(z[0]))

def average_pace_sub_6_over_time():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x = np.asarray(runs.start_date)
    y = np.asarray(runs.pace)

    filtered_x = x[y < 6]
    filtered_y = y[y < 6]

    ax1.plot_date(filtered_x, filtered_y)
    ax1.set_title('Average Pace Under 6 Over Time')

    x2 = mdates.date2num(filtered_x)
    z = np.polyfit(x2, filtered_y, 1)
    p = np.poly1d(z)
    plt.plot(filtered_x, p(x2), 'r--')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.show()
    print('Slope: ' + str(z[0]))

def average_pace_sub_530_over_time():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x = np.asarray(runs.start_date)
    y = np.asarray(runs.pace)

    filtered_x = x[y < 5.5]
    filtered_y = y[y < 5.5]

    ax1.plot_date(filtered_x, filtered_y)
    ax1.set_title('Average Pace Under 5:30 Over Time')

    x2 = mdates.date2num(filtered_x)
    z = np.polyfit(x2, filtered_y, 1)
    p = np.poly1d(z)
    plt.plot(filtered_x, p(x2), 'r--')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.show()
    print('Slope: ' + str(z[0]))

def average_pace_sub_5_over_time():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    x = np.asarray(runs.start_date)
    y = np.asarray(runs.pace)

    filtered_x = x[y < 5]
    filtered_y = y[y < 5]

    ax1.plot_date(filtered_x, filtered_y)
    ax1.set_title('Average Pace Under 5 Over Time')

    x2 = mdates.date2num(filtered_x)
    z = np.polyfit(x2, filtered_y, 1)
    p = np.poly1d(z)
    plt.plot(filtered_x, p(x2), 'r--')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.show()
    print('Slope: ' + str(z[0]))

def var_correlation():
    corr = runs.corr()
    plt.figure(figsize = (12,8))
    sns.heatmap(corr, fmt=".2f");
    plt.title('Correlation Between Dataset Variables')
    plt.show()


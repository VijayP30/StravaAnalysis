import os
from flask import Flask, render_template
import graph_generator
import folium
from folium.plugins import HeatMap
import pandas as pd
import polyline
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_route():
    map_data = generate_map()
    return render_template('map.html', map_data=map_data)

# @app.route('/graph/<graph_name>')
# def graph_route(graph_name):
#     if graph_name not in graph_data:
#         return "Graph not found", 404
#     return render_template('graph.html', graph_name=graph_name, graph_data=graph_data[graph_name])

def generate_map():
    csv_file_path = 'result/all_strava_activities.csv'

    df = pd.read_csv(csv_file_path)

    if 'map.summary_polyline' not in df.columns:
        raise ValueError("Column 'map.summary_polyline' not found in the CSV file.")
    i = 0
    while True:
        try:
            first_route_summary_polyline = df['map.summary_polyline'].iloc[i]
            cleaned_polyline = clean_polyline(first_route_summary_polyline)
            points = decode_polyline(cleaned_polyline)
            center_lat, center_lon = 39.8283,-98.5795
            my_map = folium.Map(location=[center_lat, center_lon], zoom_start=5)
            break
        except:
            i += 1

    route_points = []
    for polyline_data in df['map.summary_polyline']:
        try:
            cleaned_polyline = clean_polyline(repr(polyline_data))
            points = decode_polyline(repr(cleaned_polyline))
            if points:
                route_points.extend(points)
        except Exception as e:
            print(f"Error decoding polyline: {e}")
            continue

    HeatMap(data=route_points, radius=5, blur=3, min_opacity=0.4).add_to(my_map)

    return my_map._repr_html_()

def clean_polyline(polyline_data):
    cleaned_polyline = polyline_data.replace('\n', '').replace('\r', '')
    return cleaned_polyline

def decode_polyline(polyline_data):
    points = []
    for lat, lon in polyline.decode(polyline_data):
        if lat is not None and lon is not None:
            points.append((lat, lon))
    return points

if __name__ == '__main__':
    app.run(debug=True)

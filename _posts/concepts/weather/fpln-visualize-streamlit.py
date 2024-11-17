import streamlit as st
import folium
import geopandas as gpd

# Read the KML file
gdf = gpd.read_file('../../../data/UIAA-UNNT.kml')

# Calculate the centroid of the geometries
centroid = gdf.geometry.centroid

# Create a Folium map
# m = folium.Map(location=[gdf.centroid.y, gdf.centroid.x], zoom_start=10)
m = folium.Map(location=[centroid.y.mean(), centroid.x.mean()], zoom_start=10)


# Add the flight path to the map
folium.GeoJson(gdf, style_function=lambda x: {'color': 'blue', 'weight': 3}).add_to(m)

# Display the map
st.components.v1.html(m._repr_html_(), height=600)
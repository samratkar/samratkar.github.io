import folium
import geopandas as gpd
from llama_index.core.tools import FunctionTool
from streamlit import session_state as st

### Define the functions logic
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: The first number
        b: The second number
        
    Returns:
        The sum of a and b
    """
    return a + b

def subtract_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first.
    
    Args:
        a: The first number
        b: The second number to subtract from the first
        
    Returns:
        The result of a - b
    """
    return a - b

def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers.
    
    Args:
        a: The number to subtract from.
        b: The number to subtract.
        
    Returns:
        The result of a - b.
    """
    return (x + y) * (x + y)

def render_fpln() -> str:
    """Render and visualize the flight plan from the flight plan file uploaded..
        
    Args: none      
        
    Returns:
        A string indicating the flight plan has been visualized.
    """
    st.write("Rendering flight plan...")

    file_path_global = st.session_state.file_path
    st.write(file_path_global)
    # Read the KML file
    gdf = gpd.read_file(file_path_global)

    # Calculate the centroid of the geometries
    centroid = gdf.geometry.centroid

    # Create a Folium map
    # m = folium.Map(location=[gdf.centroid.y, gdf.centroid.x], zoom_start=10)
    m = folium.Map(location=[centroid.y.mean(), centroid.x.mean()], zoom_start=10)


    # Add the flight path to the map
    folium.GeoJson(gdf, style_function=lambda x: {'color': 'blue', 'weight': 3}).add_to(m)

    # Display the map
    st.components.v1.html(m._repr_html_(), height=600)
    return "the flight plan visualized"

# Convert these functions to LlamaIndex FunctionTools
def get_add_tool() -> FunctionTool:
    """Get the add function tool."""
    return FunctionTool.from_defaults(
        fn=add_numbers,
        name="add",
        description="Add two numbers together"
    )

def get_subtract_tool() -> FunctionTool:
    """Get the subtract function tool."""
    return FunctionTool.from_defaults(
        fn=subtract_numbers,
        name="subtract",
        description="Subtract the second number from the first"
    )

def get_mystery_tool() -> FunctionTool:
    """Get the mystery function tool.""" 
    return FunctionTool.from_defaults(
        fn=mystery,
        name="mystery",
        description="Mystery function that operates on top of two numbers"
    )

def get_render_flight_plan_tool() -> FunctionTool:
    """Gets the rendering and visualization of flight plan tool""" 
    return FunctionTool.from_defaults(
        fn=render_fpln,
        name="render flight plan",
        description="Renders a flight plan from a flight planning file. It visualizes the flight plan from source to destination"
    )

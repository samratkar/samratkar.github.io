import os
import sys
import math
import json
import csv
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Any

import requests
import numpy as np
from PIL import Image
import cv2
import torch
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import folium

try:
    import h3
except ImportError:
    h3 = None

try:
    from dotenv import load_dotenv
    # Load .env from current directory or script directory
    load_dotenv()
    script_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(script_env):
        load_dotenv(dotenv_path=script_env)
except ImportError:
    pass

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    SegformerImageProcessor = None
    SegformerForSemanticSegmentation = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SidewalkGeotagger")


class SidewalkGeotagger:
    def __init__(self, google_api_key: Optional[str] = None, model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"):
        """
        Initializes the Sidewalk Geotagger with Google Maps API access 
        and a pretrained semantic segmentation model.
        """
        self.api_key = google_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """Loads Hugging Face SegFormer model weights."""
        if SegformerImageProcessor is None or SegformerForSemanticSegmentation is None:
            logger.warning("transformers library not installed. Install via 'pip install transformers'.")
            return

        logger.info(f"Loading segmentation model '{self.model_name}' onto {self.device}...")
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def fetch_satellite_image(self, lat: float, lon: float, zoom: int = 19, size: int = 640) -> Tuple[Image.Image, Dict[str, float]]:
        """
        Fetches satellite/skyview imagery from Google Static Maps API and computes its bounding box.
        """
        if not self.api_key:
            raise ValueError(
                "A Google Maps API key is required to download static satellite tiles. "
                "Set GOOGLE_MAPS_API_KEY environment variable or pass --api-key."
            )

        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": f"{size}x{size}",
            "maptype": "satellite",
            "key": self.api_key
        }

        logger.info(f"Fetching aerial image centered at ({lat}, {lon}) with zoom {zoom}...")
        response = requests.get(url, params=params, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Google Maps API error ({response.status_code}): {response.text}")

        img = Image.open(response.raw).convert("RGB")
        bounds = self.get_mercator_bounds(lat, lon, zoom, size, size)
        return img, bounds

    def load_local_image(self, image_path: str, lat: float, lon: float, zoom: int = 19) -> Tuple[Image.Image, Dict[str, float]]:
        """
        Loads a local aerial image and computes its geographic bounding box based on center coordinates and zoom.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        bounds = self.get_mercator_bounds(lat, lon, zoom, width, height)
        logger.info(f"Loaded local image '{image_path}' ({width}x{height}) with geographic bounds: {bounds}")
        return img, bounds

    def get_mercator_bounds(self, center_lat: float, center_lon: float, zoom: int, width: int, height: int) -> Dict[str, float]:
        """
        Computes the North-South-East-West bounding box in WGS84 coordinates for the Web Mercator tile.
        """
        def lat_lon_to_world(lat: float, lon: float) -> Tuple[float, float]:
            sin_y = math.sin(lat * math.pi / 180.0)
            sin_y = min(max(sin_y, -0.9999), 0.9999)
            x = 256.0 * (0.5 + lon / 360.0)
            y = 256.0 * (0.5 - math.log((1.0 + sin_y) / (1.0 - sin_y)) / (4.0 * math.pi))
            return x, y

        def world_to_lat_lon(x: float, y: float) -> Tuple[float, float]:
            lon = (x / 256.0 - 0.5) * 360.0
            y2 = 0.5 - y / 256.0
            lat = 90.0 - 360.0 * math.atan(math.exp(-y2 * 2.0 * math.pi)) / math.pi
            return lat, lon

        scale = 1 << zoom
        world_x, world_y = lat_lon_to_world(center_lat, center_lon)

        top_left_x = world_x - (width / 2.0) / scale
        top_left_y = world_y - (height / 2.0) / scale
        bottom_right_x = world_x + (width / 2.0) / scale
        bottom_right_y = world_y + (height / 2.0) / scale

        north, west = world_to_lat_lon(top_left_x, top_left_y)
        south, east = world_to_lat_lon(bottom_right_x, bottom_right_y)

        return {"north": north, "south": south, "east": east, "west": west}

    def detect_sidewalk_mask(self, image: Image.Image, target_keywords: Optional[List[str]] = None) -> np.ndarray:
        """
        Runs semantic segmentation to detect sidewalk / pavement pixels.
        Dynamically finds target class IDs from the model's id2label configuration.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Segmentation model is not loaded.")

        if target_keywords is None:
            target_keywords = ["sidewalk", "pavement", "footpath", "footway", "road", "street", "path"]

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Upsample predictions to match original image dimensions
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Identify relevant target class IDs
        id2label = getattr(self.model.config, "id2label", {})
        target_ids = [
            cid for cid, label in id2label.items()
            if any(kw in label.lower() for kw in target_keywords)
        ]

        # Fallback if no specific labels matched
        if not target_ids:
            target_ids = [0, 1]
            logger.info(f"Using default target class IDs: {target_ids}")
        else:
            matching_labels = [f"{cid}: {id2label[cid]}" for cid in target_ids]
            logger.info(f"Identified matching target classes: {matching_labels}")

        # Log breakdown of detected top classes
        unique_classes, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        detected_summary = [
            f"'{id2label.get(cid, str(cid))}': {round(count / total_pixels * 100, 1)}%"
            for cid, count in zip(unique_classes, counts)
        ]
        logger.info(f"Class coverage breakdown: {', '.join(detected_summary)}")

        # Generate binary mask matching any of the target class IDs
        binary_mask = np.isin(pred_mask, target_ids).astype(np.uint8) * 255

        # Morphological post-processing to smooth contours and close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        return cleaned_mask

    def save_debug_visualizations(
        self,
        image: Image.Image,
        binary_mask: np.ndarray,
        aerial_path: str = "aerial_tile.png",
        overlay_path: str = "pavement_overlay.png"
    ):
        """
        Saves the raw satellite tile and an annotated overlay showing detected sidewalks in cyan.
        """
        # Save raw aerial image
        image.save(aerial_path)
        
        # Create cyan overlay on detected sidewalks
        img_np = np.array(image).copy()
        overlay = img_np.copy()
        
        # Color pavement pixels cyan [0, 220, 255]
        overlay[binary_mask > 0] = [0, 220, 255]
        
        # Blend original with overlay
        blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
        
        # Draw contour borders in blue
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 100, 255), 2)
        
        Image.fromarray(blended).save(overlay_path)
        logger.info(f"Saved aerial image to '{aerial_path}' and visual overlay to '{overlay_path}'.")

    def mask_to_geotagged_features(
        self,
        binary_mask: np.ndarray,
        bounds: Dict[str, float],
        min_area_px: int = 60,
        h3_res: int = 13
    ) -> List[Dict[str, Any]]:
        """
        Extracts contours from the binary segmentation mask and translates
        pixel coordinates into geographic coordinates (Latitude, Longitude),
        computing centroids, bounding boxes, real-world area in m², and H3 Hexagon spatial indices.
        """
        height, width = binary_mask.shape
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        north, south = bounds["north"], bounds["south"]
        west, east = bounds["east"], bounds["west"]
        if west > east:
            west, east = bounds["west"], bounds["east"]

        # Approximate meters per degree at mean latitude
        mean_lat = (north + south) / 2.0
        lat_m_per_deg = 111132.954 - 559.822 * math.cos(2 * math.radians(mean_lat)) + 1.175 * math.cos(4 * math.radians(mean_lat))
        lon_m_per_deg = 111412.84 * math.cos(math.radians(mean_lat)) - 93.5 * math.cos(3 * math.radians(mean_lat))

        # Ground Sampling Distance (meters per pixel)
        m_per_px_x = abs(east - west) * lon_m_per_deg / width
        m_per_px_y = abs(north - south) * lat_m_per_deg / height
        sq_m_per_px = m_per_px_x * m_per_px_y

        geotagged_features = []
        feature_id = 1

        for contour in contours:
            pixel_area = cv2.contourArea(contour)
            if pixel_area < min_area_px:
                continue

            # Simplify contour polygon geometry using Ramer-Douglas-Peucker (RDP)
            epsilon = 0.008 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            coords_lon_lat = []
            coords_lat_lon = []

            for pt in approx:
                px, py = pt[0]

                # Affine linear interpolation: pixel (px, py) -> (longitude, latitude)
                geo_lon = west + (px / width) * (east - west)
                geo_lat = north - (py / height) * (north - south)

                coords_lon_lat.append((geo_lon, geo_lat))
                coords_lat_lon.append((round(geo_lat, 7), round(geo_lon, 7)))

            # Ensure polygon ring is closed
            if coords_lon_lat[0] != coords_lon_lat[-1]:
                coords_lon_lat.append(coords_lon_lat[0])
                coords_lat_lon.append(coords_lat_lon[0])

            poly = Polygon(coords_lon_lat)
            if not poly.is_valid or poly.area == 0:
                continue

            # Compute centroid lat/lon
            centroid_lon, centroid_lat = poly.centroid.x, poly.centroid.y
            min_lon, min_lat, max_lon, max_lat = poly.bounds

            # Estimated real-world metrics
            area_m2 = round(pixel_area * sq_m_per_px, 2)
            perimeter_m = round(cv2.arcLength(contour, True) * ((m_per_px_x + m_per_px_y) / 2.0), 2)

            # H3 Hexagonal Spatial Indexing
            h3_centroid = None
            h3_hexagons = []
            if h3 is not None:
                try:
                    h3_centroid = h3.latlng_to_cell(centroid_lat, centroid_lon, h3_res)
                    
                    # Collect hexagon cells from polygon vertices and internal raster sampling
                    sampled_cell_ids = set()
                    if h3_centroid:
                        sampled_cell_ids.add(h3_centroid)
                    for lat_pt, lon_pt in coords_lat_lon:
                        sampled_cell_ids.add(h3.latlng_to_cell(lat_pt, lon_pt, h3_res))

                    # Sample interior points within bounding box
                    x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
                    step_x = max(1, w_box // 12)
                    step_y = max(1, h_box // 12)
                    for py_s in range(y_box, y_box + h_box, step_y):
                        for px_s in range(x_box, x_box + w_box, step_x):
                            if cv2.pointPolygonTest(contour, (float(px_s), float(py_s)), False) >= 0:
                                g_lon = west + (px_s / width) * (east - west)
                                g_lat = north - (py_s / height) * (north - south)
                                sampled_cell_ids.add(h3.latlng_to_cell(g_lat, g_lon, h3_res))

                    for cell_id in sorted(list(sampled_cell_ids)):
                        boundary_pts = h3.cell_to_boundary(cell_id) # list of (lat, lon)
                        h3_hexagons.append({
                            "h3_index": cell_id,
                            "resolution": h3_res,
                            "centroid": [round(b[0], 7) for b in [h3.cell_to_latlng(cell_id)]][0] if hasattr(h3, "cell_to_latlng") else None,
                            "boundary": [(round(b_lat, 7), round(b_lon, 7)) for b_lat, b_lon in boundary_pts]
                        })
                except Exception as ex:
                    logger.debug(f"H3 indexing warning: {ex}")

            feature = {
                "id": feature_id,
                "feature_type": "pavement",
                "centroid_lat": round(centroid_lat, 7),
                "centroid_lon": round(centroid_lon, 7),
                "h3_centroid": h3_centroid,
                "h3_hexagons": h3_hexagons,
                "h3_count": len(h3_hexagons),
                "area_sq_m": area_m2,
                "perimeter_m": perimeter_m,
                "bbox": [round(min_lat, 7), round(min_lon, 7), round(max_lat, 7), round(max_lon, 7)],
                "num_vertices": len(coords_lat_lon) - 1,
                "coordinates_lat_lon": coords_lat_lon,
                "geometry": poly
            }
            geotagged_features.append(feature)
            feature_id += 1

        logger.info(f"Extracted {len(geotagged_features)} geotagged sidewalk/pavement polygon(s).")
        return geotagged_features

    def print_geotagged_summary(self, features: List[Dict[str, Any]], verbose_coords: bool = False, verbose_hex: bool = False):
        """
        Prints a formatted tabular summary of all detected pavements with their lat/lon coordinates and H3 Hexagon IDs.
        """
        if not features:
            print("\n" + "=" * 90)
            print(" NO PAVEMENTS / SIDEWALKS DETECTED")
            print(" TIP: Try setting --target-classes sidewalk road street path or using Cityscapes model.")
            print("=" * 90 + "\n")
            return

        total_area = sum(f["area_sq_m"] for f in features)
        total_hexes = sum(f.get("h3_count", 0) for f in features)
        print("\n" + "=" * 115)
        print(f" GEOTAGGED PAVEMENTS & H3 HEXAGONS SUMMARY ({len(features)} detected | Total Area: {total_area:.1f} sq m | Total Hexagons: {total_hexes})")
        print("=" * 115)
        header = f"{'ID':<4} | {'Centroid (Lat, Lon)':<25} | {'H3 Centroid (Res 13)':<20} | {'Hex Count':<9} | {'Area (sq m)':<11} | {'Perimeter (m)':<13} | {'Vertices'}"
        print(header)
        print("-" * 115)

        for f in features:
            centroid_str = f"{f['centroid_lat']:.6f}, {f['centroid_lon']:.6f}"
            h3_cent = f.get("h3_centroid") or "N/A"
            hex_cnt = f.get("h3_count", 0)
            row = f"{f['id']:<4} | {centroid_str:<25} | {h3_cent:<20} | {hex_cnt:<9} | {f['area_sq_m']:<11.1f} | {f['perimeter_m']:<13.1f} | {f['num_vertices']}"
            print(row)

        print("=" * 115)

        if verbose_coords:
            print("\nDETAILED PAVEMENT LAT/LON VERTEX PATHS:")
            for f in features:
                print(f"\n--- Pavement #{f['id']} (Centroid: {f['centroid_lat']:.7f}, {f['centroid_lon']:.7f} | Area: {f['area_sq_m']} sq m) ---")
                for idx, (lat, lon) in enumerate(f["coordinates_lat_lon"][:-1], start=1):
                    print(f"  Point {idx:2d}: Lat {lat:.7f}, Lon {lon:.7f}")
            print("-" * 60)

        if verbose_hex or verbose_coords:
            print("\nH3 HEXAGONAL SPATIAL CELLS (H3 INDEX & BOUNDARIES):")
            for f in features:
                hexes = f.get("h3_hexagons", [])
                print(f"\n--- Pavement #{f['id']} (Covering {len(hexes)} Hexagons @ Res {hexes[0]['resolution'] if hexes else 'N/A'}) ---")
                for idx, h in enumerate(hexes, start=1):
                    print(f"  Hex #{idx:2d} [ID: {h['h3_index']}]: 6 Vertices -> {h['boundary']}")
            print("-" * 60 + "\n")

    def export_geojson(self, features: List[Dict[str, Any]], output_path: str = "detected_sidewalks.geojson"):
        """
        Saves detected features into a GeoJSON file with rich properties and standard EPSG:4326 CRS.
        """
        if not features:
            logger.warning("No features to export to GeoJSON.")
            return

        gdf = gpd.GeoDataFrame(
            [
                {
                    "id": f["id"],
                    "feature_type": f["feature_type"],
                    "centroid_lat": f["centroid_lat"],
                    "centroid_lon": f["centroid_lon"],
                    "h3_centroid": f.get("h3_centroid"),
                    "h3_hex_count": f.get("h3_count", 0),
                    "area_sq_m": f["area_sq_m"],
                    "perimeter_m": f["perimeter_m"],
                    "num_vertices": f["num_vertices"],
                    "bbox": str(f["bbox"]),
                    "geometry": f["geometry"]
                }
                for f in features
            ],
            crs="EPSG:4326"
        )
        gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"Successfully saved GeoJSON to '{output_path}'")

    def export_hexagons_geojson(self, features: List[Dict[str, Any]], output_path: str = "detected_hexagons.geojson"):
        """
        Exports all unique H3 Hexagons covering detected pavements into a dedicated GeoJSON polygon file.
        """
        if not features:
            logger.warning("No features to export hexagons for.")
            return

        hex_records = {}
        for f in features:
            for h in f.get("h3_hexagons", []):
                h_id = h["h3_index"]
                if h_id not in hex_records:
                    # Shapely Polygon expects (lon, lat)
                    poly_coords = [(lon, lat) for lat, lon in h["boundary"]]
                    if poly_coords[0] != poly_coords[-1]:
                        poly_coords.append(poly_coords[0])
                    hex_records[h_id] = {
                        "h3_index": h_id,
                        "resolution": h["resolution"],
                        "pavement_id": f["id"],
                        "geometry": Polygon(poly_coords)
                    }

        if not hex_records:
            logger.info("No H3 hexagons generated.")
            return

        gdf_hex = gpd.GeoDataFrame(list(hex_records.values()), crs="EPSG:4326")
        gdf_hex.to_file(output_path, driver="GeoJSON")
        logger.info(f"Successfully saved {len(hex_records)} H3 Hexagons to '{output_path}'")

    def export_csv(self, features: List[Dict[str, Any]], output_path: str = "detected_pavements.csv"):
        """
        Exports detected pavement metadata, H3 hexagon IDs, and coordinates into a structured CSV file.
        """
        if not features:
            logger.warning("No features to export to CSV.")
            return

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id",
                "feature_type",
                "centroid_lat",
                "centroid_lon",
                "h3_centroid",
                "h3_hex_count",
                "h3_hex_ids",
                "area_sq_m",
                "perimeter_m",
                "min_lat",
                "min_lon",
                "max_lat",
                "max_lon",
                "num_vertices",
                "coordinates_lat_lon"
            ])
            for feat in features:
                hex_ids = [h["h3_index"] for h in feat.get("h3_hexagons", [])]
                writer.writerow([
                    feat["id"],
                    feat["feature_type"],
                    feat["centroid_lat"],
                    feat["centroid_lon"],
                    feat.get("h3_centroid"),
                    len(hex_ids),
                    ";".join(hex_ids),
                    feat["area_sq_m"],
                    feat["perimeter_m"],
                    feat["bbox"][0],
                    feat["bbox"][1],
                    feat["bbox"][2],
                    feat["bbox"][3],
                    feat["num_vertices"],
                    json.dumps(feat["coordinates_lat_lon"])
                ])
        logger.info(f"Successfully saved CSV summary to '{output_path}'")

    def export_json(self, features: List[Dict[str, Any]], output_path: str = "detected_pavements.json"):
        """
        Exports detected pavement metadata, H3 hexagon structures, and lat/lon coordinate arrays into a JSON file.
        """
        if not features:
            logger.warning("No features to export to JSON.")
            return

        export_data = []
        for feat in features:
            export_data.append({
                "id": feat["id"],
                "feature_type": feat["feature_type"],
                "centroid": {
                    "lat": feat["centroid_lat"],
                    "lon": feat["centroid_lon"]
                },
                "h3_spatial_index": {
                    "centroid_h3_index": feat.get("h3_centroid"),
                    "hexagon_count": feat.get("h3_count", 0),
                    "hexagons": feat.get("h3_hexagons", [])
                },
                "area_sq_m": feat["area_sq_m"],
                "perimeter_m": feat["perimeter_m"],
                "bbox": {
                    "min_lat": feat["bbox"][0],
                    "min_lon": feat["bbox"][1],
                    "max_lat": feat["bbox"][2],
                    "max_lon": feat["bbox"][3]
                },
                "num_vertices": feat["num_vertices"],
                "coordinates": [
                    {"lat": pt[0], "lon": pt[1]} for pt in feat["coordinates_lat_lon"]
                ]
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"total_features": len(export_data), "pavements": export_data}, f, indent=2)
        logger.info(f"Successfully saved JSON to '{output_path}'")

    def generate_interactive_map(
        self,
        features: List[Dict[str, Any]],
        center_lat: float,
        center_lon: float,
        output_html: str = "sidewalk_map.html"
    ):
        """
        Generates an interactive Folium web map visualizing the detected sidewalk polygons
        alongside H3 Hexagonal Grid overlays with layer toggles.
        """
        m = folium.Map(location=[center_lat, center_lon], zoom_start=19, tiles="OpenStreetMap")

        # Layer for Sidewalk Polygons
        sidewalk_group = folium.FeatureGroup(name="Detected Sidewalks / Pavements", show=True)
        # Layer for H3 Hexagonal Grid
        hex_group = folium.FeatureGroup(name="H3 Hexagonal Grid (Amber)", show=True)
        # Layer for Centroid Markers
        marker_group = folium.FeatureGroup(name="Centroid Markers", show=True)

        for f in features:
            coords = [(lat, lon) for lat, lon in f["coordinates_lat_lon"]]
            h3_cent = f.get("h3_centroid") or "N/A"
            hex_cnt = f.get("h3_count", 0)

            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
                <h4 style="margin: 0 0 5px 0; color: #0074D9;">Pavement #{f['id']}</h4>
                <b>Centroid Lat:</b> {f['centroid_lat']:.6f}<br/>
                <b>Centroid Lon:</b> {f['centroid_lon']:.6f}<br/>
                <b>H3 Index:</b> <code>{h3_cent}</code><br/>
                <b>Hexagons:</b> {hex_cnt} cells<br/>
                <b>Area:</b> {f['area_sq_m']} sq m<br/>
                <b>Perimeter:</b> {f['perimeter_m']} m<br/>
                <b>Vertices:</b> {f['num_vertices']}
            </div>
            """
            folium.Polygon(
                locations=coords,
                color="#0074D9",
                weight=3,
                fill=True,
                fill_color="#7FDBFF",
                fill_opacity=0.5,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"Pavement #{f['id']}: ({f['centroid_lat']:.5f}, {f['centroid_lon']:.5f})"
            ).add_to(sidewalk_group)

            # Draw H3 Hexagons
            for h in f.get("h3_hexagons", []):
                h_boundary = [(lat, lon) for lat, lon in h["boundary"]]
                h_popup = f"""
                <div style="font-family: sans-serif; font-size: 11px; width: 190px;">
                    <h5 style="margin: 0 0 4px 0; color: #FF851B;">H3 Hexagon</h5>
                    <b>H3 Cell ID:</b> <code>{h['h3_index']}</code><br/>
                    <b>Resolution:</b> {h['resolution']}<br/>
                    <b>Pavement ID:</b> #{f['id']}
                </div>
                """
                folium.Polygon(
                    locations=h_boundary,
                    color="#FF851B",
                    weight=1.5,
                    fill=True,
                    fill_color="#FFDC00",
                    fill_opacity=0.35,
                    popup=folium.Popup(h_popup, max_width=220),
                    tooltip=f"H3 Hex: {h['h3_index']}"
                ).add_to(hex_group)

            # Add circle marker at each pavement centroid
            folium.CircleMarker(
                location=[f["centroid_lat"], f["centroid_lon"]],
                radius=4,
                color="#001f3f",
                fill=True,
                fill_color="#0074D9",
                fill_opacity=0.9,
                tooltip=f"Centroid #{f['id']}: {f['centroid_lat']:.6f}, {f['centroid_lon']:.6f}"
            ).add_to(marker_group)

        # Mark query center
        folium.Marker(
            [center_lat, center_lon],
            tooltip="Center Query Location",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(marker_group)

        sidewalk_group.add_to(m)
        hex_group.add_to(m)
        marker_group.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        m.save(output_html)
        logger.info(f"Successfully saved interactive visualization to '{output_html}'")


def main():
    parser = argparse.ArgumentParser(description="Detect and Geotag Sidewalks / Pavements with H3 Hexagons from Aerial Imagery")
    parser.add_argument("--lat", type=float, default=37.7749, help="Center latitude (e.g. 37.7749)")
    parser.add_argument("--lon", type=float, default=-122.4194, help="Center longitude (e.g. -122.4194)")
    parser.add_argument("--zoom", type=int, default=19, help="Google Maps zoom level (default: 19)")
    parser.add_argument("--size", type=int, default=640, help="Tile size in pixels (default: 640)")
    parser.add_argument("--image-path", type=str, default=None, help="Path to local aerial image (skips API call if provided)")
    parser.add_argument("--api-key", type=str, default=None, help="Google Maps API Key (or set GOOGLE_MAPS_API_KEY env)")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", help="HuggingFace SegFormer model (default: cityscapes-1024-1024)")
    parser.add_argument("--target-classes", nargs="+", default=["sidewalk", "pavement", "footpath", "footway", "road", "street", "path"], help="Classes to detect")
    parser.add_argument("--min-area-px", type=int, default=60, help="Minimum contour area in pixels (default: 60)")
    parser.add_argument("--h3-res", type=int, default=13, help="H3 Hexagon resolution (default: 13, ~3.5m edge)")
    parser.add_argument("--verbose-coords", action="store_true", help="Print all individual vertex lat/lon coordinates in console")
    parser.add_argument("--verbose-hex", action="store_true", help="Print all H3 Hexagon IDs and 6-point boundary coordinates")
    parser.add_argument("--geojson-out", type=str, default="detected_sidewalks.geojson", help="Output GeoJSON path")
    parser.add_argument("--hex-geojson-out", type=str, default="detected_hexagons.geojson", help="Output H3 Hexagons GeoJSON path")
    parser.add_argument("--csv-out", type=str, default="detected_pavements.csv", help="Output CSV summary path")
    parser.add_argument("--json-out", type=str, default="detected_pavements.json", help="Output JSON path")
    parser.add_argument("--map-out", type=str, default="sidewalk_map.html", help="Output Folium HTML map path")

    args = parser.parse_args()

    geotagger = SidewalkGeotagger(google_api_key=args.api_key, model_name=args.model)

    try:
        # 1. Fetch satellite / skyview tile or load local image
        if args.image_path:
            image, bounds = geotagger.load_local_image(
                image_path=args.image_path, lat=args.lat, lon=args.lon, zoom=args.zoom
            )
        else:
            image, bounds = geotagger.fetch_satellite_image(
                lat=args.lat, lon=args.lon, zoom=args.zoom, size=args.size
            )

        # 2. Detect sidewalk / pavement mask
        mask = geotagger.detect_sidewalk_mask(image, target_keywords=args.target_classes)

        # 3. Save visual debugging images (raw tile + overlay)
        geotagger.save_debug_visualizations(image, mask)

        # 4. Geotag mask pixels to real-world WGS84 polygons with H3 Hexagons
        features = geotagger.mask_to_geotagged_features(
            mask, bounds, min_area_px=args.min_area_px, h3_res=args.h3_res
        )

        # 5. Output geotagged lat/lon report with H3 hexagons to console
        geotagger.print_geotagged_summary(
            features, verbose_coords=args.verbose_coords, verbose_hex=args.verbose_hex
        )

        # 6. Export results in multiple formats
        if args.geojson_out:
            geotagger.export_geojson(features, output_path=args.geojson_out)
        if args.hex_geojson_out:
            geotagger.export_hexagons_geojson(features, output_path=args.hex_geojson_out)
        if args.csv_out:
            geotagger.export_csv(features, output_path=args.csv_out)
        if args.json_out:
            geotagger.export_json(features, output_path=args.json_out)
        if args.map_out:
            geotagger.generate_interactive_map(
                features, center_lat=args.lat, center_lon=args.lon, output_html=args.map_out
            )

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


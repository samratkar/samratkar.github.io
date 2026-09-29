"""Lab 3 - the planner-facing application (Streamlit).

The app calls the SAME /predict endpoint a backend would. It does not load the
model and it does not touch Feast. That is the design rule worth defending: if
the app loaded its own copy of the model, FreshMart would have two serving paths
to keep in step, two places to roll back, and two answers to the same question.

Run: streamlit run serving/planner_app.py
"""
import pandas as pd, requests, streamlit as st

st.set_page_config(page_title="FreshMart Reorder Planner", layout="wide")
st.title("FreshMart - Reorder Propensity Planner")

API = st.sidebar.text_input("API base URL", "http://localhost:8000")
try:
    r = requests.get(f"{API}/ready", timeout=3).json()
    st.sidebar.success(f"model v{r.get('model_version')} ({r.get('alias')})")
    st.sidebar.caption(f"{r.get('model_type')} · data {r.get('data_hash')}")
except Exception:
    st.sidebar.error("service not ready - start the API first")

st.subheader("Score one store-product line")
c1, c2 = st.columns(2)
store = c1.text_input("store_id", "S001")
product = c2.text_input("product_id", "P0067")

if st.button("Predict"):
    try:
        resp = requests.post(f"{API}/predict",
                             json={"store_id": store, "product_id": product}, timeout=10)
        if resp.status_code == 200:
            d = resp.json()
            m1, m2, m3 = st.columns(3)
            m1.metric("Reorder probability", f"{d['reorder_probability']:.1%}")
            m2.metric("Recommended action", d["action"].title())
            m3.metric("Model version", d["model_version"])
        elif resp.status_code == 404:
            # the unknown-key path from Lab 1, surfaced to a human in words they
            # can act on rather than as a stack trace
            st.warning(f"No features for {store}/{product}. Either the line is new "
                       f"and has never sold, or the id is wrong.")
        else:
            st.error(f"{resp.status_code}: {resp.text}")
    except requests.RequestException as exc:
        st.error(f"could not reach the API: {exc}")

st.divider()
st.subheader("Morning run - score a whole store")
store_bulk = st.text_input("store_id for the batch", "S001")
n = st.slider("lines to score", 10, 200, 50, step=10)
if st.button("Run batch"):
    lines = (pd.read_parquet("feature_repo/data/reorder_features.parquet")
               .query("store_id == @store_bulk")[["product_id"]]
               .drop_duplicates().head(n))
    rows = []
    for pid in lines.product_id:
        resp = requests.post(f"{API}/predict",
                             json={"store_id": store_bulk, "product_id": pid}, timeout=10)
        if resp.status_code == 200:
            rows.append(resp.json())
    if rows:
        out = (pd.DataFrame(rows)[["product_id", "reorder_probability", "action"]]
                 .sort_values("reorder_probability", ascending=False))
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.caption(f"{(out.action != 'no action').sum()} of {len(out)} lines need attention. "
                   f"For thousands of lines use the batch scorer, not this loop.")
    else:
        st.info("nothing scored - is the API running?")

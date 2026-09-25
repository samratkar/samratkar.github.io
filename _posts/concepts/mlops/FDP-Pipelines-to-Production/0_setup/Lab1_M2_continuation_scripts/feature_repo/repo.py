"""Feast feature repository - the feature definitions.

The entities are the grain M2 handed over: a store and a product. The source is
the parquet the feature-engineering stage writes, and the FeatureView lists what
is served. `feast apply` registers these; `feast materialize` loads the online
store so a planner's request can be answered by entity key in milliseconds.

One definition, two paths: get_historical_features for training (point-in-time
correct), get_online_features for serving. That single definition is what keeps
Lab 2's training rows and Lab 3's serving requests describing the same world.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
import os

BASE = os.path.dirname(__file__)

store   = Entity(name="store",   join_keys=["store_id"])
product = Entity(name="product", join_keys=["product_id"])

source = FileSource(
    path=os.path.join(BASE, "data", "reorder_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

reorder_fv = FeatureView(
    name="reorder_features",
    entities=[store, product],
    ttl=timedelta(days=365),
    online=True,
    source=source,
    schema=[
        Field(name="units_7d",             dtype=Int64),
        Field(name="units_28d",            dtype=Int64),
        Field(name="trend_ratio",          dtype=Float32),
        Field(name="days_since_last_sale", dtype=Int64),
        Field(name="quiet_days_28d",       dtype=Int64),
        Field(name="avg_unit_price_7d",    dtype=Float32),
        Field(name="lead_time_days",       dtype=Int64),
        Field(name="promo_active",         dtype=Int64),
    ],
)

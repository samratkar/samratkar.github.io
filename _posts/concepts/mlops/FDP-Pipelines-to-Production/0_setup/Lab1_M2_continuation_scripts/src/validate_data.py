"""Lab 1 / stage: validate - the DATA CONTRACT, enforced with Great Expectations,
and a five-pillars data-observability snapshot.

The contract is not invented here. It is M2's handoff, made executable: the grain
is Store x Product x Day, and the quality rules the analysis said the pipeline
must guarantee were "no missing quantity, no duplicate IDs, prices in range".
Each of those is one expectation below. An ExpectationSuite is a named,
versionable artifact the whole team can read and diff - which is the DataOps way
to make a producer/consumer agreement enforceable rather than remembered.

We also compute the five pillars of data observability (freshness, volume,
schema, distribution, lineage) so the pipeline emits an observability record on
every run, not only a pass/fail.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import great_expectations as gx

EXPECTED_SCHEMA = {
    "transaction_id": "object", "transaction_date": "datetime64[ns]",
    "store_id": "object", "product_id": "object",
    "quantity": "int", "unit_price": "float", "revenue": "float",
}
PRICE_MIN, PRICE_MAX = 20.0, 500.0


def build_suite(ctx):
    suite = gx.ExpectationSuite(name="pos_contract")
    E = gx.expectations
    for col in EXPECTED_SCHEMA:                                  # 7 - the agreed schema
        suite.add_expectation(E.ExpectColumnToExist(column=col))
    # M2's rule 1: no duplicate IDs
    suite.add_expectation(E.ExpectColumnValuesToBeUnique(column="transaction_id"))
    # M2's rule 2: no missing quantity, and a quantity is a positive count
    suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="quantity"))
    suite.add_expectation(E.ExpectColumnValuesToBeBetween(column="quantity", min_value=1, max_value=50))
    # M2's rule 3: prices in range
    suite.add_expectation(E.ExpectColumnValuesToBeBetween(column="unit_price",
                                                          min_value=PRICE_MIN, max_value=PRICE_MAX))
    # the grain itself must be intact, or nothing downstream can group by it
    suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="store_id"))
    suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="product_id"))
    suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column="transaction_date"))
    suite.add_expectation(E.ExpectColumnValuesToBeBetween(column="revenue", min_value=0, max_value=100_000))
    suite.add_expectation(E.ExpectTableRowCountToBeBetween(min_value=1000, max_value=10_000_000))
    return suite


def five_pillars(df: pd.DataFrame) -> dict:
    """Freshness, Volume, Schema, Distribution, Lineage - the five pillars."""
    schema_ok = all(k in df.columns for k in EXPECTED_SCHEMA)
    return {
        "freshness":    {"latest_event": str(df["transaction_date"].max().date()),
                         "span_days": int((df["transaction_date"].max() -
                                           df["transaction_date"].min()).days)},
        "volume":       {"rows": int(len(df)),
                         "store_product_pairs": int(df.groupby(["store_id", "product_id"]).ngroups)},
        "schema":       {"expected": list(EXPECTED_SCHEMA), "matches": bool(schema_ok)},
        "distribution": {"mean_basket": round(float(df.quantity.mean()), 3),
                         "mean_unit_price": round(float(df.unit_price.mean()), 2)},
        "lineage":      {"data_hash": Path("data/pos_raw.hash").read_text().strip()
                         if Path("data/pos_raw.hash").exists() else "unknown"},
    }


def validate(inp, report_path):
    df = pd.read_parquet(inp)
    ctx = gx.get_context(mode="ephemeral")
    bd = (ctx.data_sources.add_pandas("freshmart")
          .add_dataframe_asset("pos")
          .add_batch_definition_whole_dataframe("batch"))
    batch = bd.get_batch(batch_parameters={"dataframe": df})
    suite = ctx.suites.add(build_suite(ctx))
    res = batch.validate(suite)

    passed = sum(1 for r in res.results if r.success)
    report = {
        "success": bool(res.success),
        "passed": passed, "total": len(res.results),
        "failed_expectations": [r.expectation_config.type for r in res.results if not r.success],
        "observability": five_pillars(df),
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pos_clean.parquet")
    ap.add_argument("--report", default="data/contract_report.json")
    a = ap.parse_args()
    rep = validate(a.inp, a.report)
    status = "PASS" if rep["success"] else "FAIL"
    print(f"[validate] POS data contract {status}  ({rep['passed']}/{rep['total']} expectations) -> {a.report}")
    obs = rep["observability"]
    print(f"[validate] observability: volume={obs['volume']['rows']:,} rows / "
          f"{obs['volume']['store_product_pairs']:,} store-product pairs | "
          f"freshness={obs['freshness']['latest_event']} ({obs['freshness']['span_days']}d span) | "
          f"schema_ok={obs['schema']['matches']} | basket={obs['distribution']['mean_basket']}")
    if not rep["success"]:
        print(f"           FAILED: {rep['failed_expectations']}")
        sys.exit(1)


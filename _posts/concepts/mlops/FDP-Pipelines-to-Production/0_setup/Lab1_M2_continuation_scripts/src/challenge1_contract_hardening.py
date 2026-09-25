# CHALLENGE solution: grow the contract past M2's three rules, and make it fail
# cleanly instead of crashing.
#   (a) validate the SCHEMA FIRST, so a missing column is reported as a contract
#       result rather than an AttributeError inside the observability code;
#   (b) referential integrity - a store_id or product_id that is not in the
#       master is a broken grain, and every group-by downstream silently drops it;
#   (c) internal consistency - revenue must equal quantity x unit_price. M2 never
#       said so because M2 only read the columns; a pipeline has to trust them.
import json, sys
from pathlib import Path
import pandas as pd, great_expectations as gx
from validate_data import EXPECTED_SCHEMA, build_suite, five_pillars

def build_hardened_suite(ctx, stores, products):
    suite = build_suite(ctx)
    E = gx.expectations
    suite.add_expectation(E.ExpectColumnValuesToBeInSet(
        column="store_id", value_set=sorted(stores.store_id.tolist())))
    suite.add_expectation(E.ExpectColumnValuesToBeInSet(
        column="product_id", value_set=sorted(products.product_id.tolist())))
    suite.add_expectation(E.ExpectColumnPairValuesToBeEqual(
        column_A="revenue", column_B="revenue_check"))
    return suite

def validate(inp, report_path="data/contract_report.json"):
    df = pd.read_parquet(inp)
    missing = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    if missing:                                    # short-circuit: no pillars yet
        report = {"success": False, "passed": 0, "total": 0,
                  "failed_expectations": [f"ExpectColumnToExist({c})" for c in missing],
                  "observability": {"schema": {"matches": False, "missing": missing}}}
        Path(report_path).write_text(json.dumps(report, indent=2))
        print(f"[validate] SCHEMA FAIL - missing {missing}; pillars not computed")
        return report

    df = df.assign(revenue_check=(df.quantity * df.unit_price).round(2))
    stores   = pd.read_parquet("data/store_master.parquet")
    products = pd.read_parquet("data/product_master.parquet")

    ctx = gx.get_context(mode="ephemeral")
    bd = (ctx.data_sources.add_pandas("freshmart").add_dataframe_asset("pos")
            .add_batch_definition_whole_dataframe("batch"))
    res = bd.get_batch(batch_parameters={"dataframe": df}).validate(
        ctx.suites.add(build_hardened_suite(ctx, stores, products)))
    report = {"success": bool(res.success),
              "passed": sum(1 for r in res.results if r.success), "total": len(res.results),
              "failed_expectations": [r.expectation_config.type for r in res.results if not r.success],
              "observability": five_pillars(df)}
    Path(report_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"[validate] hardened contract {'PASS' if res.success else 'FAIL'} "
          f"({report['passed']}/{report['total']} expectations)")
    if not res.success:
        print(f"           FAILED: {report['failed_expectations']}")
    return report

if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "data/pos_clean.parquet",
             "data/hardened_report.json")

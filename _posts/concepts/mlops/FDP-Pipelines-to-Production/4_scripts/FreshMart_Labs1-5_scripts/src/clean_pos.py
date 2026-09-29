"""Lab 1 / stage: clean - execute M2's triage decisions, once, upstream.

M2's Mission 1 challenge routed each quality finding to ENGINEERING or to local
analysis. The three findings routed to Engineering land here, and nowhere else:

  duplicate transaction_id   -> drop the re-sent rows (the feed retries at 06:10)
  missing quantity           -> DROP the row, do not impute. M2's profile-triage
                                said so explicitly: a missing core measure is not
                                a value to invent, and a silently imputed median
                                becomes a feature the model learns from.
  unit_price out of range    -> quarantine, do not clip; a price of 0 or 50,000 is
                                a feed defect, and clipping hides it from the report.

Cleaning is separate from validation on purpose. This stage makes the data
conform; the next stage checks that it did. If the same script both fixed and
approved the data, the contract would be marking its own homework.
"""
import argparse, json
from pathlib import Path
import pandas as pd

PRICE_MIN, PRICE_MAX = 20.0, 500.0


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    df = df.drop_duplicates(subset="transaction_id", keep="first")
    n_dupes = n0 - len(df)

    n1 = len(df)
    df = df[df.quantity.notna() & (df.quantity > 0)]
    n_missing = n1 - len(df)

    n2 = len(df)
    quarantine = df[(df.unit_price < PRICE_MIN) | (df.unit_price > PRICE_MAX)]
    df = df.drop(index=quarantine.index)
    n_price = n2 - len(df)

    df = df.assign(quantity=df.quantity.astype("int64"))
    df["revenue"] = (df.quantity * df.unit_price).round(2)      # recompute, never trust
    return df.reset_index(drop=True), {
        "rows_in": n0, "rows_out": len(df), "dropped_duplicate_id": n_dupes,
        "dropped_missing_quantity": n_missing, "quarantined_price": n_price}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pos_raw.parquet")
    ap.add_argument("--out", default="data/pos_clean.parquet")
    ap.add_argument("--report", default="data/clean_report.json")
    a = ap.parse_args()

    out, rep = clean(pd.read_parquet(a.inp))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    Path(a.report).write_text(json.dumps(rep, indent=2))
    print(f"[clean] {rep['rows_in']:,} -> {rep['rows_out']:,} rows  "
          f"(-{rep['dropped_duplicate_id']} duplicate ids, "
          f"-{rep['dropped_missing_quantity']} missing quantity, "
          f"-{rep['quarantined_price']} price out of range)")

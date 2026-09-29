"""Lab 1 / fixtures - the deliberately broken batches the challenges run against.

A contract is only convincing when you watch it fail, so the missions need input
that fails in one specific way each. Rather than corrupt data by hand (and get a
different result in every room), build the broken batches from the clean one,
deterministically, and keep them out of the DVC pipeline: these are teaching
fixtures, not stages.

Each batch breaks exactly one clause, which is the point - a student should be
able to predict which expectation fires before running anything.

  pos_badstore.parquet   40 rows relabelled to a store that is not in the master
                         -> referential integrity (Mission 1 challenge)
  pos_badrev.parquet     25 rows with revenue inflated 18% against quantity x price
                         -> internal consistency (Mission 1 challenge)
  pos_noqty.parquet      the quantity column dropped entirely
                         -> schema, and the crash the ordering bug causes
  pos_dupes.parquet      150 transactions re-sent, on top of an already-clean batch
                         -> uniqueness, if you want the duplicate demo after cleaning
"""
import argparse
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pos_clean.parquet")
    ap.add_argument("--outdir", default="data/broken")
    a = ap.parse_args()

    df = pd.read_parquet(a.inp)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    bad_store = df.copy()
    bad_store.loc[bad_store.index[:40], "store_id"] = "S099"       # not in store_master
    bad_store.to_parquet(out / "pos_badstore.parquet", index=False)

    bad_rev = df.copy()
    bad_rev.loc[bad_rev.index[:25], "revenue"] = (
        bad_rev.loc[bad_rev.index[:25], "revenue"] * 1.18).round(2)
    bad_rev.to_parquet(out / "pos_badrev.parquet", index=False)

    df.drop(columns=["quantity"]).to_parquet(out / "pos_noqty.parquet", index=False)

    pd.concat([df, df.head(150)], ignore_index=True).to_parquet(
        out / "pos_dupes.parquet", index=False)

    print(f"[fixtures] 4 broken batches -> {out}/  "
          f"(badstore, badrev, noqty, dupes) from {len(df):,} clean rows")

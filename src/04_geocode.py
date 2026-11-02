"""03_geocode.py

Extracted from docs/docs/dataset.md. Edit the guide and re-extract,
or edit here and keep the guide in step.

    python src/03_geocode.py
"""

import os
import pandas as pd
import reverse_geocoder as rg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "workouts.csv")


def main():
    df = pd.read_csv(PATH)
    res = rg.search(list(zip(df.start_lat, df.start_lon)))   # offline, fast in batch

    df["city"] = [r["name"] for r in res]
    df["country"] = [r["cc"] for r in res]
    df.to_csv(PATH, index=False)

    print(df.country.value_counts().head(10))
    print(df.city.value_counts().head(15))


# reverse_geocoder spawns worker processes. On Windows each worker re-imports the
# main module, so without this guard the script restarts itself recursively.
if __name__ == "__main__":
    main()

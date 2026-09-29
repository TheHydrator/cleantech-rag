import os
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()
CSV_PATH = "data/cleantech_media_dataset_v3_2024-10-28.csv"


def clean(value):
    """Turn pandas' empty values (NaN) into None, which Postgres understands as empty."""
    return None if pd.isna(value) else value


# Read the CSV and rename the ID column
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"Unnamed: 0": "id"})
df["published"] = pd.to_datetime(df["date"], errors="coerce").dt.date

# Check for problems BEFORE touching the database
missing_titles = df["title"].isna().sum()
duplicate_ids = df["id"].duplicated().sum()
bad_dates = df["published"].isna().sum()
print(f"Rows: {len(df)}, missing titles: {missing_titles}, "
      f"duplicate ids: {duplicate_ids}, unparseable dates: {bad_dates}")

if missing_titles or duplicate_ids:
    raise SystemExit("Stopping: fix the data problems above before loading.")

# Load into Postgres
rows = [
    (int(r.id), r.title, clean(r.published), clean(r.domain), clean(r.url))
    for r in df.itertuples()
]

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO articles (id, title, published, domain, url)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (id) DO NOTHING""",
            rows,
        )
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

print(f"Articles in database: {count}")
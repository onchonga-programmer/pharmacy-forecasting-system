import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.database_config import DB_CONFIG


class PharmacySalesETL:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    # -------------------- DATABASE --------------------
    def connect(self):
        return psycopg2.connect(**DB_CONFIG)

    # -------------------- EXTRACT --------------------
    def extract(self):
        print(f"Reading CSV: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Parse datum column and extract Year/Week
        self.df['datum'] = pd.to_datetime(self.df['datum'], format='%m/%d/%Y')
        self.df['Year'] = self.df['datum'].dt.isocalendar().year
        self.df['Week'] = self.df['datum'].dt.isocalendar().week
        
        print(f"✓ Loaded {len(self.df)} rows")

    # -------------------- LOAD DRUGS --------------------
    def load_drugs(self):
        print("\n=== Loading Drugs Dimension ===")

        drug_columns = [
            col for col in self.df.columns
            if col not in ['datum', 'Year', 'Week']
        ]

        drugs_data = []
        for drug in drug_columns:
            atc_code = drug[:5]  # assumption documented
            drugs_data.append((drug, atc_code))

        query = """
        INSERT INTO raw_data.drugs_raw (drug_name, atc_code)
        VALUES (%s, %s)
        ON CONFLICT (drug_name, atc_code) DO NOTHING
        """

        conn = self.connect()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, query, drugs_data, page_size=500)
            conn.commit()
            print(f"✓ {len(drugs_data)} drugs processed")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return drug_columns

    # -------------------- TRANSFORM --------------------
    def transform_sales(self, drug_columns):
        print("\n=== Transforming Sales Data (Wide → Long) ===")

        long_df = self.df.melt(
            id_vars=['Year', 'Week'],
            value_vars=drug_columns,
            var_name='drug_name',
            value_name='quantity'
        )

        long_df = long_df.dropna()
        long_df = long_df[long_df['quantity'] > 0]

        # ISO week-safe dates
        long_df['week_start_date'] = pd.to_datetime(
            long_df['Year'].astype(str)
            + '-W'
            + long_df['Week'].astype(str).str.zfill(2)
            + '-1',
            format='%G-W%V-%u',
            errors='coerce'
        )

        long_df = long_df.dropna(subset=['week_start_date'])
        long_df['week_end_date'] = long_df['week_start_date'] + pd.Timedelta(days=6)
        long_df['avg_daily_quantity'] = (long_df['quantity'] / 7).round(2)

        print(f"✓ Transformed to {len(long_df)} rows")
        return long_df

    # -------------------- LOAD FACT TABLE --------------------
    def load_sales(self, sales_df: pd.DataFrame):
        print("\n=== Loading Weekly Sales Fact Table ===")

        conn = self.connect()

        try:
            with conn.cursor() as cur:
                # Fixed: drug_name should be the key, drug_id the value
                cur.execute("SELECT drug_name, drug_id FROM raw_data.drugs_raw")
                drug_map = dict(cur.fetchall())

            # Debug: Print drug names comparison
            print(f"Drug map from DB: {list(drug_map.keys())}")
            print(f"Unique drugs in sales_df: {sales_df['drug_name'].unique().tolist()}")
            
            sales_df['drug_id'] = sales_df['drug_name'].map(drug_map)
            
            # Debug: Check how many matched
            matched = sales_df['drug_id'].notna().sum()
            print(f"Matched {matched} out of {len(sales_df)} rows")
            
            sales_df = sales_df.dropna(subset=['drug_id'])

            records = [
                (
                    int(r.Year),
                    int(r.Week),
                    r.week_start_date.date(),
                    r.week_end_date.date(),
                    int(r.drug_id),
                    int(r.quantity),
                    float(r.avg_daily_quantity)
                )
                for r in sales_df.itertuples(index=False)
            ]

            query = """
            INSERT INTO raw_data.sales_weekly_raw
            (year, week_number, week_start_date, week_end_date,
             drug_id, total_quantity, avg_daily_quantity)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (year, week_number, drug_id) DO UPDATE
            SET total_quantity = EXCLUDED.total_quantity,
                avg_daily_quantity = EXCLUDED.avg_daily_quantity
            """

            with conn.cursor() as cur:
                execute_batch(cur, query, records, page_size=1000)

            conn.commit()
            print(f"✓ Inserted {len(records)} sales records")

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -------------------- VERIFY --------------------
    def verify(self):
        print("\n=== Verifying Data ===")

        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MIN(week_start_date), MAX(week_start_date)
                FROM raw_data.sales_weekly_raw
            """)
            start, end = cur.fetchone()
            print(f"Date range: {start} → {end}")

            cur.execute("SELECT COUNT(*) FROM raw_data.sales_weekly_raw")
            print(f"Total rows: {cur.fetchone()[0]:,}")

            cur.execute("""
                SELECT d.drug_name, SUM(sw.total_quantity)
                FROM raw_data.sales_weekly_raw sw
                JOIN raw_data.drugs_raw d ON sw.drug_id = d.drug_id
                GROUP BY d.drug_name
                ORDER BY SUM(sw.total_quantity) DESC
                LIMIT 5
            """)
            print("\nTop 5 drugs:")
            for drug, qty in cur.fetchall():
                print(f"  {drug}: {qty:,}")

        conn.close()

    # -------------------- RUN PIPELINE --------------------
    def run(self):
        self.extract()
        drug_cols = self.load_drugs()
        sales_df = self.transform_sales(drug_cols)
        self.load_sales(sales_df)
        self.verify()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    csv_file = "data/raw/salesweekly.csv"

    if not Path(csv_file).exists():
        raise FileNotFoundError(f"{csv_file} not found")

    etl = PharmacySalesETL(csv_file)
    etl.run()

    print("\n✅ ETL PIPELINE WAS COMPLETED SUCCESSFULLY")

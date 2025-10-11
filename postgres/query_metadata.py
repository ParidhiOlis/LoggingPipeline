from sqlalchemy import create_engine, MetaData, Table, text
import pandas as pd

# Create engine
engine = create_engine("postgresql+psycopg2://app_user:supersecret@localhost:5432/metadata_db")

# Create metadata object
metadata = MetaData()

# Reflect the documents table
table = Table('documents', metadata, schema='core', autoload_with=engine)

# Run query
with engine.connect() as conn:
    query = text("SELECT id, source_type FROM core.documents;")
    result = conn.execute(query)
    print("Result object:", result)
    
    results_as_dict = []
    results_manual = []
    for row in result:

        print("Row id:", row.id)  # Print row for debugging
        results_as_dict.append(row._asdict())  # Convert to dict using _asdict()
        results_manual.append({column: value for column, value in zip(result.keys(), row)})  # Manual conversion

    df = pd.read_sql(query, con=engine)
    print(df.shape)


print("Using _asdict():", results_as_dict)
print("Manual conversion:", results_manual)


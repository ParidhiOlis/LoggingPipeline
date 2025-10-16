import psycopg2

conn = conn = psycopg2.connect(
    dbname="metadata_db",
    user="app_user",
    password="supersecret",
    host="localhost",   # container name if script runs in another container
    port=5432
)
cur = conn.cursor()

# remove all rows first
cur.execute("TRUNCATE TABLE core.documents;")

# TODO: Update the acl information to make it free of conflicts between allowed and denied users
with open("./postgres/file_metadata.csv", "r", encoding="utf-8") as f:
    cur.copy_expert("""
        COPY core.documents (id,object_key,owner_id,filename,source_type,created_at,updated_at,confidentiality,acl,status)
        FROM STDIN WITH (FORMAT csv, HEADER true)
                    
    """, f)
conn.commit()
cur.close(); conn.close()

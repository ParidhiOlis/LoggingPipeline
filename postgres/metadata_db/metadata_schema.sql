-- Runs on first startup of a fresh data dir.

-- 1) Make sure this matches POSTGRES_DB in your docker-compose:
\connect metadata_db;

-- 2) Create schema owned by your app user (ensure app_user exists as POSTGRES_USER)
CREATE SCHEMA IF NOT EXISTS core;

-- 3) Create table *in the core schema* (explicit schema-qualification)
CREATE TABLE IF NOT EXISTS core.documents (
  id            SERIAL PRIMARY KEY,
  object_key    TEXT UNIQUE NOT NULL,   -- MinIO object key; likely unique
  owner_id      TEXT NOT NULL,
  source_type   TEXT NOT NULL CHECK (source_type IN ('docs', 'slack', 'email')),
  filename      TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  confidentiality TEXT NOT NULL,        -- "Public|Internal|Confidential|Restricted"
  acl           JSONB NOT NULL,         -- {"allow_roles":[],"allow_groups":[],"allow_users":[],"deny":[]}
  status        TEXT NOT NULL DEFAULT 'NEW' -- NEW|INGESTED|ERROR
);

-- 4) Useful indexes (PK already has an index; don't re-index id)
CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON core.documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON core.documents(status);
-- object_key is UNIQUE above, which creates an index automatically

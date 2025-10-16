import os
import re
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# from sqlalchemy import (
#     create_engine, MetaData, Table, Column, String, Boolean, DateTime, Text
# )
# from sqlalchemy.engine import Engine
# from sqlalchemy.dialects.postgresql import VARCHAR
# from sqlalchemy.exc import IntegrityError

# --------------------
# Config & Setup
# --------------------
load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
# DATABASE_URL = os.environ["DATABASE_URL"]
EXPORT_DIR = Path(os.environ.get("SLACK_EXPORT_DIR", "./data/slack_exports")).resolve()
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

client = WebClient(token=SLACK_BOT_TOKEN)

# engine: Engine = create_engine(DATABASE_URL, future=True)
# metadata = MetaData()

# --------------------
# DB Schema (idempotent)
# --------------------
# slack_users = Table(
#     "slack_users", metadata,
#     Column("user_id", String(64), primary_key=True),
#     Column("team_id", String(64)),
#     Column("name", String(255)),
#     Column("real_name", String(255)),
#     Column("email", String(255)),
#     Column("is_bot", Boolean),
#     Column("updated_at", DateTime, default=datetime.now(timezone.utc)),
# )

# slack_channels = Table(
#     "slack_channels", metadata,
#     Column("channel_id", String(64), primary_key=True),
#     Column("name", String(255)),
#     Column("type", String(32)),          # public_channel | private_channel | im | mpim
#     Column("is_private", Boolean),
#     Column("is_im", Boolean),
#     Column("is_mpim", Boolean),
#     Column("team_id", String(64)),
#     Column("created", DateTime, nullable=True),
#     Column("archived", Boolean, default=False),
#     Column("updated_at", DateTime, default=datetime.now(timezone.utc)),
# )

# slack_channel_members = Table(
#     "slack_channel_members", metadata,
#     Column("channel_id", String(64)),
#     Column("user_id", String(64)),
#     Column("updated_at", DateTime, default=datetime.now(timezone.utc)),
# )

# slack_messages = Table(
#     "slack_messages", metadata,
#     Column("ts", String(32), primary_key=True),  # Slack message ts is unique per channel, but we store globally for simplicity
#     Column("channel_id", String(64), index=True),
#     Column("user_id", String(64), index=True, nullable=True),
#     Column("text", Text),
#     Column("thread_ts", String(32), nullable=True),
#     Column("subtype", String(64), nullable=True),
#     Column("is_bot", Boolean),
#     Column("is_app_user", Boolean),
#     Column("has_files", Boolean),
#     Column("inserted_at", DateTime, default=datetime.now(timezone.utc)),
# )

# metadata.create_all(engine)

# --------------------
# Utilities
# --------------------
def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", name)
    return name or "untitled"

def channel_display_name(ch: dict, users_by_id: Dict[str, dict]) -> str:
    """
    Create a readable name for DM/MPIM; otherwise use channel name.
    """
    if ch.get("is_im"):
        uid = ch.get("user")
        u = users_by_id.get(uid, {})
        return f"DM__{u.get('real_name') or u.get('name') or uid}"
    if ch.get("is_mpim"):
        # group DM: join member names if available
        member_names = []
        for uid in ch.get("members", []):
            u = users_by_id.get(uid, {})
            member_names.append(u.get("real_name") or u.get("name") or uid)
        if member_names:
            return "GDM__" + "_".join(sanitize_filename(n) for n in member_names)
        return ch.get("name") or ch.get("id")
    # channels
    return ch.get("name") or ch.get("id")

def write_csv(path: Path, header: List[str], rows: List[Tuple]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

# def upsert(conn, table: Table, values: dict, key_fields: List[str]):
#     """
#     Simple upsert: try insert; on conflict, do a delete+insert fallback
#     (portable without PostgreSQL-specific ON CONFLICT). For speed, you may
#     replace this with PostgreSQL 'ON CONFLICT DO UPDATE' if desired.
#     """
#     try:
#         conn.execute(table.insert().values(**values))
#     except IntegrityError:
#         conn.rollback()
#         # naive update: delete then insert
#         del_cond = {k: values[k] for k in key_fields}
#         conn.execute(table.delete().where(
#             *(table.c[k] == v for k, v in del_cond.items())
#         ))
#         conn.execute(table.insert().values(**values))

# --------------------
# Slack Fetchers
# --------------------
def fetch_all_users() -> Dict[str, dict]:
    users = {}
    cursor = None
    while True:
        resp = client.users_list(cursor=cursor, limit=200)
        for u in resp["members"]:
            users[u["id"]] = u
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return users

def fetch_all_conversations() -> List[dict]:
    """
    Fetch public channels, private channels, IMs, MPIMs.
    """
    convs = []
    cursor = None
    # Slack lets multiple types in one call separated by commas
    types = "public_channel,private_channel,im,mpim"
    while True:
        resp = client.conversations_list(types=types, cursor=cursor, limit=1000)
        convs.extend(resp["channels"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return convs

def fetch_members(channel_id: str) -> List[str]:
    members = []
    cursor = None
    while True:
        try:
            resp = client.conversations_members(channel=channel_id, cursor=cursor, limit=1000)
        except SlackApiError as e:
            # For IM/MPIM or restricted channels, members may not be available
            if e.response["error"] in {"not_in_channel", "channel_not_found", "method_not_supported_for_channel_type"}:
                break
            raise
        members.extend(resp.get("members", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return members

def fetch_history(channel_id: str):
    """
    Generator yielding messages from oldest to newest (Slack returns newest first).
    """
    cursor = None
    first_page = True
    while True:
        resp = client.conversations_history(channel=channel_id, cursor=cursor, limit=1000, oldest="0")
        msgs = resp.get("messages", [])
        for m in reversed(msgs):  # oldest first
            yield m
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        # Be gentle with rate limits
        if not first_page:
            time.sleep(0.5)
        first_page = False

# --------------------
# Main ingest
# --------------------
def main():
    print("Fetching users…")
    users_by_id = fetch_all_users()
    print(f"Users: {len(users_by_id)}")

    print("Fetching conversations…")
    conversations = fetch_all_conversations()
    print(f"Conversations: {len(conversations)}")

    # Add members list for im/mpim if missing (Slack often includes for mpim, not for im)
    for ch in conversations:
        # print(ch)
        if ch.get("is_im"):
            # For IM, slack includes 'user' (the other party). Compose members = [me, user]
            # We can't easily know 'me' id from bot; skip, but later RBAC can use messages.user_id
            ch["members"] = [ch.get("user")] if ch.get("user") else []
        elif ch.get("is_mpim") or ch.get("is_private") or (not ch.get("is_im") and not ch.get("is_mpim")):
            # Try to fetch members for channels we can query
            try:
                ch["members"] = fetch_members(ch["id"])
            except SlackApiError:
                ch["members"] = []


    
#     # Prepare CSV accumulators
#     channels_csv_rows = []
#     users_csv_rows = []
#     members_csv_rows = []
#     messages_csv_rows = []

#     # Export users to DB & CSV
#     with engine.begin() as conn:
#         for uid, u in users_by_id.items():
#             values = {
#                 "user_id": uid,
#                 "team_id": (u.get("team_id") or u.get("profile", {}).get("team")) or "",
#                 "name": u.get("name") or "",
#                 "real_name": u.get("real_name") or "",
#                 "email": u.get("profile", {}).get("email") or "",
#                 "is_bot": bool(u.get("is_bot", False)),
#                 "updated_at": datetime.now(timezone.utc),
#             }
#             upsert(conn, slack_users, values, ["user_id"])
#             users_csv_rows.append((
#                 values["user_id"], values["team_id"], values["name"], values["real_name"],
#                 values["email"], values["is_bot"], values["updated_at"].isoformat()
#     #         ))

#     # Process each conversation: write TXT, store metadata, messages
    for ch in conversations:
        ch_id = ch["id"]
        ch_type = (
            "im" if ch.get("is_im")
            else "mpim" if ch.get("is_mpim")
            else "private_channel" if ch.get("is_private")
            else "public_channel"
        )
        disp_name = channel_display_name(ch, users_by_id)
        fname = sanitize_filename(disp_name) + ".txt"
        fpath = EXPORT_DIR / fname

        created_ts = ch.get("created")
        created_dt = datetime.fromtimestamp(created_ts, tz=timezone.utc) if created_ts else None

#         # Upsert channel row
#         with engine.begin() as conn:
#             ch_values = {
#                 "channel_id": ch_id,
#                 "name": ch.get("name") or disp_name,
#                 "type": ch_type,
#                 "is_private": bool(ch.get("is_private", False)),
#                 "is_im": bool(ch.get("is_im", False)),
#                 "is_mpim": bool(ch.get("is_mpim", False)),
#                 "team_id": ch.get("context_team_id") or ch.get("shared_team_ids", [None])[0] or "",
#                 "created": created_dt,
#                 "archived": bool(ch.get("is_archived", False)),
#                 "updated_at": datetime.now(timezone.utc),
#             }
#             upsert(conn, slack_channels, ch_values, ["channel_id"])

#         channels_csv_rows.append((
#             ch_id, ch_values["name"], ch_values["type"], ch_values["is_private"],
#             ch_values["is_im"], ch_values["is_mpim"], ch_values["team_id"],
#             ch_values["created"].isoformat() if ch_values["created"] else "",
#             ch_values["archived"], ch_values["updated_at"].isoformat()
#         ))

#         # Upsert channel members (RBAC base)
#         with engine.begin() as conn:
#             member_ids = ch.get("members") or []
#             for uid in member_ids:
#                 upsert(conn, slack_channel_members, {
#                     "channel_id": ch_id,
#                     "user_id": uid,
#                     "updated_at": datetime.now(timezone.utc),
#                 }, ["channel_id", "user_id"])
#                 members_csv_rows.append((
#                     ch_id, uid, datetime.now(timezone.utc).isoformat()
#                 ))

#         # Write transcript and collect messages metadata
        lines = []
        msg_count = 0
        for m in fetch_history(ch_id):
            ts = float(m["ts"])
            user_id = m.get("user") or m.get("bot_id") or None
            user_name = ""
            if user_id and user_id in users_by_id:
                u = users_by_id[user_id]
                user_name = u.get("real_name") or u.get("name") or user_id
            elif m.get("username"):  # apps/bots sometimes set 'username'
                user_name = m["username"]
            else:
                user_name = user_id or "unknown"

            text = m.get("text", "")
            thread_ts = m.get("thread_ts")
            subtype = m.get("subtype")
            is_bot = bool(m.get("bot_id"))
            is_app_user = bool(m.get("app_id"))
            has_files = bool(m.get("files"))

#             # Human-friendly line
            lines.append(f"{iso(ts)} | {user_name}: {text}".rstrip())
            msg_count += 1

#             # Persist message metadata to DB
#             with engine.begin() as conn:
#                 upsert(conn, slack_messages, {
#                     "ts": m["ts"],
#                     "channel_id": ch_id,
#                     "user_id": m.get("user"),
#                     "text": text,
#                     "thread_ts": thread_ts,
#                     "subtype": subtype,
#                     "is_bot": is_bot,
#                     "is_app_user": is_app_user,
#                     "has_files": has_files,
#                     "inserted_at": datetime.now(timezone.utc),
#                 }, ["ts"])

#             # For CSV export
#             messages_csv_rows.append((
#                 m["ts"], ch_id, m.get("user") or "", text.replace("\n", " "),
#                 thread_ts or "", subtype or "", is_bot, is_app_user, has_files
#             ))

#         # Write the transcript file
        header = [
            f"# Slack Export",
            f"# Name: {disp_name}",
            f"# Channel ID: {ch_id}",
            f"# Type: {ch_type}, Private: {bool(ch.get('is_private', False))}, IM: {bool(ch.get('is_im', False))}, MPIM: {bool(ch.get('is_mpim', False))}",
            f"# Members: {', '.join(ch.get('members', []))}",
            f"# Messages: {msg_count}",
            f"# Exported at: {datetime.now(timezone.utc).isoformat()}",
            "#" * 60,
        ]
        with fpath.open("w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for line in lines:
                f.write(line + "\n")

        print(f"Wrote {msg_count:5d} msgs → {fpath.name}")

#     # Write CSVs
#     write_csv(EXPORT_DIR / "users.csv",
#               ["user_id", "team_id", "name", "real_name", "email", "is_bot", "updated_at"],
#               users_csv_rows)

#     write_csv(EXPORT_DIR / "channels.csv",
#               ["channel_id", "name", "type", "is_private", "is_im", "is_mpim", "team_id", "created", "archived", "updated_at"],
#               channels_csv_rows)

#     write_csv(EXPORT_DIR / "channel_members.csv",
#               ["channel_id", "user_id", "updated_at"],
#               members_csv_rows)

#     write_csv(EXPORT_DIR / "messages_metadata.csv",
#               ["ts", "channel_id", "user_id", "text", "thread_ts", "subtype", "is_bot", "is_app_user", "has_files"],
#               messages_csv_rows)

#     print(f"\nDone. Exports in: {EXPORT_DIR}")

if __name__ == "__main__":
    main()

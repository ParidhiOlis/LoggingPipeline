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

if __name__ == "__main__":
    main()

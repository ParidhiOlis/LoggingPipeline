# file: redis_smoke_test.py
import json, time
from typing import Dict, List, TypedDict
import redis


class Profile(TypedDict):
    username: str
    current_roles: List[str]
    current_groups: List[str]

# --- Connect ---
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
APP, ENV = "rag", "prod"
def K(*parts): return ":".join((APP, ENV, *map(str, parts)))

# ---- 1) USER PROFILE ----
def save_user_profile(uid: str, profile: Dict[str, str]) -> None:
    r.hset(K("user", uid, "profile"), mapping=profile)

def get_user_profile(uid: str) -> Dict[str, str]:
    return r.hgetall(K("user", uid, "profile"))

# ---- 2) CONVERSATION MEMORY (summary + last N messages) ----
MAX_TURNS = 100

def persist_memory(uid: str, messages: List[dict]) -> None:
    if messages:
        pipe = r.pipeline()
        for m in messages:
            pipe.rpush(K("user", uid, "messages"), json.dumps(m))
        pipe.ltrim(K("user", uid, "messages"), -MAX_TURNS, -1)
        pipe.execute()

def load_memory(uid: str):
    raw = r.lrange(K("user", uid, "messages"), 0, -1)
    msgs = [json.loads(x) for x in raw]
    return msgs

# ---- 3) RECENT DOCS (ZSET + HASH per doc) ----
def add_recent_doc(uid: str, doc_id: str, name: str, location: str) -> None:
    r.hset(K("doc", doc_id), mapping={"name": name, "location": location})
    r.zadd(K("user", uid, "recent_docs"), {doc_id: int(time.time())})

def list_recent_docs(uid: str, limit: int = 10):
    ids = r.zrevrange(K("user", uid, "recent_docs"), 0, limit - 1)
    return [{"doc_id": d, **r.hgetall(K("doc", d))} for d in ids]

# -------- Smoke run --------
if __name__ == "__main__":
    uid= "suhaask"

    # print("Saving profile…")
    # save_user_profile(uid, {
    #         'username': 'suhaask',
    #         'current_roles': json.dumps(['swe']), 
    #         'current_groups': json.dumps(['product', '_ALL_']), 
    #     })
    # print("Profile:", get_user_profile(uid))

    # print("Persisting memory…")
    # persist_memory(
    #     uid,
    #     # summary_text="User asked about invoices; assistant summarized prior steps.",
    #     messages=[
    #     ],
    # )
    msgs = load_memory(uid)
    print("Messages:", msgs)

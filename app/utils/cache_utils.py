
import json
import redis
from app import config
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory
)

# --------- Start: User cache utils ----------
class UserCache:
    def __init__(self, host, port):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.APP, self.ENV = "rag", "prod"

    def K(self, *parts): 
        return ":".join((self.APP, self.ENV, *map(str, parts)))

    def get_memory_from_cache(self,
        uid: str,
        memory: ConversationBufferWindowMemory,
        fetch_last: int = config.MEMORY_MAX_TURNS,
    ) -> ConversationBufferWindowMemory:
        """
        Load messages from Redis (list of JSON dicts) and stuff them into CBWM.
        CBWM will apply its own window (k exchanges) when you call .load_memory_variables().
        """
        print("uid - ", uid)
        key = self.K("user", uid, "messages")
        print("key - ", key)
        raw = self.r.lrange(key, max(0, -fetch_last), -1)  # last N only
        print("raw - ", raw)
        if not raw:
            memory.chat_memory.messages = []
        else:
            msg_dicts = [json.loads(x) for x in raw]
            lc_messages = messages_from_dict(msg_dicts)
            # Store the full history; CBWM filters to last k exchanges when used.
            memory.chat_memory.messages = lc_messages
        return memory

    def persist_memory_to_cache(self,
        uid: str,
        memory: ConversationBufferWindowMemory,
        max_turns: int = config.USER_CACHE_MAX_TURNS,
    ) -> None:
        """
        Persist the full message list currently held by CBWM into Redis.
        We overwrite with the current in-memory sequence and trim to `max_turns`.
        """
        key = self.K("user", uid, "messages")
        print("\n\nkey after - ", key)
        print("memory - ", memory)
        msg_dicts = messages_to_dict(memory.chat_memory.messages)
        pipe = self.r.pipeline()
        pipe.delete(key)
        # push in order
        for d in msg_dicts:
            pipe.rpush(key, json.dumps(d))
        pipe.ltrim(key, -max_turns, -1)     # Limit the number of messages stored
        pipe.execute()

# --------- End: User cache utils ----------

from __future__ import annotations

import json
import sqlite3
import argparse
from pathlib import Path

from cryptography.fernet import Fernet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    selected = {int(item) for item in args.ids.split(",") if item.strip()}
    root = Path(__file__).resolve().parents[1]
    decrypt = Fernet((root / "api" / "chat_secret.key").read_bytes()).decrypt
    connection = sqlite3.connect(f"file:{root / 'api' / 'chats.db'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT id, chat_id, role, content_enc, meta_json FROM chat_messages ORDER BY id"
    ).fetchall()
    by_chat: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_chat.setdefault(row["chat_id"], []).append(row)
    pairs = []
    for messages in by_chat.values():
        for user, assistant in zip(messages, messages[1:]):
            if user["role"] == "user" and assistant["role"] == "assistant":
                pairs.append((user, assistant))
    print(f"pairs={len(pairs)}")
    for user, assistant in pairs:
        if selected and int(assistant["id"]) not in selected:
            continue
        question = decrypt(user["content_enc"].encode()).decode("utf-8", errors="replace")
        answer = decrypt(assistant["content_enc"].encode()).decode("utf-8", errors="replace")
        metadata = json.loads(assistant["meta_json"] or "{}")
        print(
            f"PAIR {user['id']}/{assistant['id']} mode={metadata.get('mode')} "
            f"Q={question[:160].replace(chr(10), ' ')} "
            f"A={(answer if args.full else answer[:360]).replace(chr(10), ' ')}"
        )


if __name__ == "__main__":
    main()

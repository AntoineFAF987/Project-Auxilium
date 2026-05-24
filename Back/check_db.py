import sqlite3
conn = sqlite3.connect('api/app.db')
print("=== Tables ===")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(tables)
print(f"\n=== Chats (total: {conn.execute('SELECT COUNT(*) FROM chats').fetchone()[0]}) ===")
for row in conn.execute("SELECT id, tenant_id, user_id, title, created_at FROM chats LIMIT 10").fetchall():
    print(row)
print(f"\n=== Messages (total: {conn.execute('SELECT COUNT(*) FROM chat_messages').fetchone()[0]}) ===")
for row in conn.execute("SELECT id, chat_id, role, created_at FROM chat_messages LIMIT 10").fetchall():
    print(row)
conn.close()

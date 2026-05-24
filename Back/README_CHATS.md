# Chats persistants (On‑Prem)

Ce backend stocke désormais les conversations de manière sécurisée, par utilisateur et par tenant (Microsoft Entra ID), dans `Back/api/app.db` (SQLite, WAL). Le contenu des messages est CHIFFRÉ au repos via Fernet (bibliothèque `cryptography`).

## Où sont stockées les données ?
- Base: `Back/api/app.db`
- Clé de chiffrement: par défaut `Back/api/chat_secret.key` (générée si `CHAT_ENC_KEY` n'est pas fourni)
- Tables: `chats`, `chat_messages`

## Clé de chiffrement (recommandé en production)
Fournir une clé Fernet via la variable d'environnement `CHAT_ENC_KEY` (urlsafe base64, 32 octets). Exemple de génération Python:

```python
from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())
```

Sous Windows PowerShell (session du service):

```powershell
$env:CHAT_ENC_KEY = "<ta_cle_fernet>"
```

Si `CHAT_ENC_KEY` est absent, une clé sera créée dans `Back/api/chat_secret.key`.

## Endpoints API
- `GET /chats` – liste les chats de l'utilisateur courant (requiert Authorization: Bearer <access_token>)
- `POST /chats` – crée un chat `{ title?: string }` → `{ id, title }`
- `GET /chats/{chat_id}/messages` – liste les messages (décryptés)
- `POST /chats/{chat_id}/messages` – ajoute un message `{ role, content, meta? }`
- `PATCH /chats/{chat_id}` – renomme `{ title }`
- `DELETE /chats/{chat_id}` – suppression logique

Tous ces endpoints sont multi‑tenant/multi‑utilisateur (isolation par `tenant_id` + `user_id`).

## Intégration avec /ask
- Si l'appel `/ask`/`/ask/stream` est accompagné d'un header `Authorization` valide:
  - Le backend crée (si besoin) la conversation (titre = 1ère question tronquée)
  - Persiste le message utilisateur et la réponse assistant
  - Retourne `chat_id` dans la réponse (et dans l'événement `done` du stream)

Le front peut réutiliser `chat_id` comme `thread_id` pour les tours suivants.

## Sauvegarde / Restauration
- Sauvegarder `Back/api/app.db` et la clé de chiffrement (env `CHAT_ENC_KEY` ou `Back/api/chat_secret.key`).
- Sans la clé, les messages chiffrés ne sont pas lisibles.

## Notes sécurité
- Chiffrement applicatif au repos (Fernet). Pense à protéger l'accès au serveur (NTFS, pare‑feu) et à journaliser/monitorer l'accès aux fichiers.
- Pour une base plus robuste (PostgreSQL avec TDE/KeyVault), la couche d'accès reste simple à porter (SQLAlchemy ou psycopg2), tout en conservant le chiffrement applicatif.

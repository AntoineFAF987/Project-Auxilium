# Guide d'installation du STREAMING

## ✅ C'est facilement réalisable !

Le streaming est maintenant implémenté. Voici ce qui a été fait et comment l'activer :

## 📦 Fichiers créés

1. **Backend** : `Back/rag_core/llm_stream.py` - Version streaming de la fonction LLM
2. **Backend** : Nouvel endpoint `/ask/stream` ajouté dans `Back/api/routes_ask.py`
3. **Frontend** : Référence de code dans `Front/src/app/STREAMING_SENDMESSAGE.tsx`

## 🔧 Étapes pour activer le streaming

### Option 1 : Remplacement manuel (RECOMMANDÉ - 5 minutes)

#### Frontend (page.tsx)

1. Ouvrez `Front/src/app/page.tsx`
2. Trouvez la fonction `sendMessage()` (vers ligne 850)
3. Remplacez-la par le contenu du fichier `STREAMING_SENDMESSAGE.tsx` que j'ai créé

**Astuce** : Cherchez cette ligne dans page.tsx :
```typescript
async function sendMessage() {
  const q = input.trim();
```

Et remplacez TOUTE la fonction (jusqu'au dernier `}` de cette fonction) par le code dans `STREAMING_SENDMESSAGE.tsx`.

#### Backend - Rien à faire !

Le backend est déjà configuré ✅
- Le fichier `llm_stream.py` est créé
- L'endpoint `/ask/stream` est ajouté à `routes_ask.py`
- L'import est déjà fait

### Option 2 : Mode hybride (TEST facile)

Si vous voulez tester avant de tout remplacer :

1. **Créez un bouton toggle** dans le frontend pour basculer entre mode normal et streaming
2. Ajoutez un état : `const [useStreaming, setUseStreaming] = useState(false);`
3. Changez l'URL conditionnellement :
```typescript
const endpoint = useStreaming ? '/ask/stream' : '/ask';
const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${endpoint}`, ...)
```

## 🚀 Redémarrage des serveurs

Après avoir fait les modifications :

### Backend
```powershell
# Si le serveur tourne, arrêtez-le (Ctrl+C) puis :
cd Back
python api/server.py
```

### Frontend
```powershell
# Si le serveur tourne, arrêtez-le (Ctrl+C) puis :
cd Front
npm run dev
```

## ✨ Résultat attendu

Après activation, vous verrez :
- Les mots s'afficher progressivement au lieu d'apparaître d'un coup
- Un effet similaire à ChatGPT, Claude, etc.
- L'utilisateur peut voir la réponse se construire en temps réel
- Le bouton "Stop" fonctionne toujours pour interrompre

## 🐛 Si ça ne marche pas

### Vérifications :

1. **Backend** : Vérifiez que l'import est bon dans `routes_ask.py` (ligne ~22) :
```python
from rag_core.llm_stream import ask_mistral_with_context_stream
```

2. **Frontend** : Vérifiez que l'URL pointe vers `/ask/stream` et non `/ask`

3. **Console** : Regardez la console navigateur (F12) pour voir s'il y a des erreurs

4. **Logs backend** : Regardez les logs du serveur Python pour voir si l'endpoint est appelé

## 📝 Notes techniques

- **Format** : SSE (Server-Sent Events) - standard pour le streaming unidirectionnel
- **Compatibilité** : Fonctionne avec tous les navigateurs modernes
- **Performance** : Très léger, pas de WebSocket nécessaire
- **Fallback** : Si le streaming échoue, l'ancien endpoint `/ask` fonctionne toujours

## 💡 Conseil

Commencez par l'Option 1 (remplacement complet). C'est plus simple et c'est ce que font ChatGPT et les autres IA modernes. L'ancien mode "tout d'un coup" n'est plus vraiment nécessaire une fois que le streaming fonctionne.

---

**Temps estimé d'installation : 5-10 minutes**
**Difficulté : Facile** ⭐⭐☆☆☆

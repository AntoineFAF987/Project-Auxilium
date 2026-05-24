# 🎯 GUIDE ULTRA-RAPIDE - Activation du Streaming

## Étape unique : Modifier page.tsx

### 📍 Localisation précise

1. Ouvrez le fichier : `Front/src/app/page.tsx`
2. Allez à la **ligne 850** (ou cherchez `async function sendMessage()`)
3. La fonction commence par :
```typescript
  async function sendMessage() {
    const q = input.trim();
    if (!q || loading) return;
```

4. Et se termine à la **ligne 974** par :
```typescript
      );
    }
  }
```

### ✂️ Supprimer

**Supprimez les lignes 850 à 974** (toute la fonction `sendMessage`)

### ➕ Remplacer par

Copiez-collez le contenu du fichier `STREAMING_SENDMESSAGE.tsx` à la place.

**OU** utilisez ce code directement :

<details>
<summary>📄 Cliquez pour voir le code complet</summary>

```typescript
  async function sendMessage() {
    const q = input.trim();
    if (!q || loading) return;

    const newUserMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: q,
      replyTo: replyTarget
        ? {
            id: replyTarget.id,
            role: replyTarget.role,
            content: replyTarget.content.slice(0, 500),
          }
        : undefined,
    };

    const nextUser = [...messages, newUserMsg];
    setMessages(nextUser);
    setInput("");
    setReplyTarget(null);
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    // Message assistant temporaire pour le streaming
    const tempAssistantId = crypto.randomUUID();
    const tempAssistant: Message = {
      id: tempAssistantId,
      role: "assistant",
      content: "",
      sources: [],
    };

    try {
      const { accessToken } = await getTokens();

      // ID de conversation pour l'API (créé si absent)
      const tid = chatId ?? (() => {
        const v = crypto.randomUUID();
        setChatId(v);
        return v;
      })();

      // 1) Historique "récent" (comme avant) — utile pour le ton du dialogue
      const historyRecent = buildHistoryPayload(messages);

      // 2) Historique ANCRÉ autour du message cible (si on a cliqué "Répondre")
      const reply_history = replyTarget ? buildReplyHistory(messages, replyTarget) : undefined;

      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          q,
          history: historyRecent,
          reply_history,
          thread_id: tid,
          source_mode: sourceMode,
          reply_to: newUserMsg.replyTo
            ? {
                id: newUserMsg.replyTo.id,
                role: newUserMsg.replyTo.role,
                content: newUserMsg.replyTo.content,
              }
            : null,
        }),
        signal: controller.signal,
      });

      if (!r.ok) {
        const errText = await r.text();
        const next = [
          ...nextUser,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `⚠️ ${errText || r.statusText || "Erreur API"}`,
          } as Message,
        ];
        setMessages(next);
        persistCurrent(next);
        setLoading(false);
        return;
      }

      // Streaming avec SSE
      const reader = r.body?.getReader();
      const decoder = new TextDecoder();
      
      let accumulatedContent = "";
      let finalSources: Source[] = [];
      let finalMode: string | undefined = undefined;

      // Ajouter le message temporaire
      setMessages([...nextUser, tempAssistant]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue;
          
          const data = line.slice(6); // Enlever "data: "
          try {
            const parsed = JSON.parse(data);
            
            if (parsed.type === 'content') {
              accumulatedContent += parsed.content;
              
              // Mettre à jour le message en temps réel
              setMessages((prev) => {
                const updated = [...prev];
                const idx = updated.findIndex((m) => m.id === tempAssistantId);
                if (idx >= 0) {
                  updated[idx] = {
                    ...updated[idx],
                    content: accumulatedContent,
                  };
                }
                return updated;
              });
              
              // Auto-scroll pendant le streaming
              requestAnimationFrame(() =>
                bottomRef.current?.scrollIntoView({ behavior: "smooth" })
              );
            } else if (parsed.type === 'done') {
              finalSources = Array.isArray(parsed.sources) ? parsed.sources : [];
              finalMode = typeof parsed.mode === 'string' ? parsed.mode : undefined;
            } else if (parsed.type === 'error') {
              throw new Error(parsed.error || 'Erreur inconnue');
            }
          } catch (e) {
            console.error('Erreur parsing SSE:', e);
          }
        }
      }

      // Finaliser le message
      const finalMessage: Message = {
        id: tempAssistantId,
        role: "assistant",
        content: accumulatedContent,
        sources: finalSources,
        mode: finalMode,
      };

      const finalMessages = [...nextUser, finalMessage];
      setMessages(finalMessages);
      persistCurrent(finalMessages);

    } catch (e: any) {
      if (e?.name === "AbortError") {
        // Annulation demandée par l'utilisateur : message court
        const next = [
          ...nextUser,
          { id: crypto.randomUUID(), role: "assistant", content: "Réflexion interrompue." } as Message,
        ];
        setMessages(next);
        persistCurrent(next);
      } else {
        const next = [
          ...nextUser,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `⚠️ Erreur réseau: ${String(e)}`,
          } as Message,
        ];
        setMessages(next);
        persistCurrent(next);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
      requestAnimationFrame(() =>
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })
      );
    }
  }
```

</details>

### 💾 Sauvegarder

Enregistrez le fichier `page.tsx`.

### 🔄 Redémarrer le frontend

```powershell
# Dans le terminal Front :
# Arrêter avec Ctrl+C si nécessaire
npm run dev
```

## ✅ C'est tout !

Le backend est déjà prêt, il suffit de modifier cette seule fonction.

## 🎬 Test

1. Posez une question à votre IA
2. Vous devriez voir les mots apparaître progressivement
3. Comme ChatGPT ! 🎉

---

**Temps total : 2 minutes** ⚡

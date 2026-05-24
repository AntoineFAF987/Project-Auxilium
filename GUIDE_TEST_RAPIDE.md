# 🚀 Guide de Test Rapide - IA Chaleureuse

## Démarrage Rapide

### 1. Redémarrer le Backend

```powershell
cd Back
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

### 2. Tester via l'Interface Frontend

#### Test Jovial
Dans l'interface de chat, envoyer :
```
Salut ! Super application ! 😄 J'adore vraiment ce que vous faites !! 🎉
```

**Résultat attendu** : Réponse enthousiaste avec emojis

---

#### Test Neutre
```
Bonjour, comment fonctionne l'indexation des documents ?
```

**Résultat attendu** : Réponse professionnelle et chaleureuse, emojis subtils ou absents

---

#### Test Formel
```
Bonjour, pourriez-vous s'il vous plaît m'expliquer la procédure ?
```

**Résultat attendu** : Réponse professionnelle avec vouvoiement, sans emojis

---

### 3. Tester via API (cURL)

```powershell
# Test Jovial
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method POST -Headers @{"Content-Type"="application/json"} -Body (@{
    q = "Salut ! Super app ! 😄 J'adore vraiment ça !! 🎉"
    history = @()
    thread_id = "test-jovial"
} | ConvertTo-Json)

# Test Neutre
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method POST -Headers @{"Content-Type"="application/json"} -Body (@{
    q = "Bonjour, comment fonctionne l'application ?"
    history = @()
    thread_id = "test-neutre"
} | ConvertTo-Json)

# Test Formel
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method POST -Headers @{"Content-Type"="application/json"} -Body (@{
    q = "Bonjour, pourriez-vous s'il vous plaît m'aider ?"
    history = @()
    thread_id = "test-formel"
} | ConvertTo-Json)
```

---

## 🔍 Points de Vérification

### ✅ Checklist de Test

- [ ] **Détection joviale fonctionne** : L'IA utilise des emojis quand l'utilisateur est enthousiaste
- [ ] **Ton professionnel maintenu** : Pas d'emojis inappropriés dans les contextes formels
- [ ] **Adaptation progressive** : Le ton change naturellement au fil de la conversation
- [ ] **Précision préservée** : Les réponses factuelles restent exactes
- [ ] **Multilingue** : Fonctionne en français ET en anglais
- [ ] **Historique** : La détection prend en compte les messages précédents

---

## 📊 Logs de Débogage

Pour voir la détection en action, ajouter temporairement dans `llm.py` après la ligne 57 :

```python
print(f"[DEBUG] is_jovial={is_jovial}, is_casual={is_casual}, is_formal={is_formal}")
```

Cela affichera dans la console :
```
[DEBUG] is_jovial=True, is_casual=True, is_formal=False
```

---

## 🎯 Tests Spécifiques

### Test 1 : Conversation Continue
1. "Salut ! 😊"
2. "Comment ça va ?"
3. "Super ! Peux-tu m'aider ?"

→ Vérifier que le ton jovial se maintient

### Test 2 : Changement de Ton
1. "Bonjour Monsieur, pourriez-vous m'aider ?"
2. "Ok merci"
3. "Cool !! 😄"

→ Vérifier la transition formel → neutre → jovial

### Test 3 : Small Talk
1. "Salut ! 👋"
2. "Ça va ?"
3. "Top ! 🎉"

→ Vérifier les réponses courtes et amicales

---

## 🐛 Dépannage

### L'IA n'utilise pas d'emojis
- Vérifier que le message contient au moins 2 marqueurs jovials
- Vérifier l'historique (derniers 3 messages)
- S'assurer que la température du modèle est appropriée

### L'IA utilise trop d'emojis
- Réduire le nombre de marqueurs jovials dans le message
- Utiliser un ton plus neutre

### Les réponses sont incohérentes
- Vider l'historique de conversation
- Vérifier le `thread_id` utilisé
- Redémarrer le backend

---

## 📞 Support

Pour toute question ou problème :
1. Vérifier les logs du backend
2. Consulter `TESTS_IA_CHALEUREUSE.md` pour les cas d'usage détaillés
3. Consulter `CHANGELOG_IA_CHALEUREUSE.md` pour les détails techniques

---

**Date** : 15 octobre 2025  
**Version** : 1.0

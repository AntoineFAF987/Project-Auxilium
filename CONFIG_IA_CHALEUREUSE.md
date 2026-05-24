# ⚙️ Configuration & Personnalisation - IA Chaleureuse

## 🎛️ Paramètres Ajustables

### 1. Marqueurs Jovials

**Fichier** : `Back/rag_core/llm.py` (ligne ~55)

```python
jovial_markers = [
    # Emojis positifs
    "😊", "😄", "😃", "🙂", "👍", "😁", "🎉", "✨",
    # Ponctuation
    "!",
    # Mots-clés français
    "haha", "lol", "cool", "super", "génial", "top", "nickel", "excellent"
]
```

#### 💡 Personnalisation

**Ajouter des emojis régionaux** :
```python
# Ajouter après "✨"
"🤩", "💪", "🔥", "👏", "🙌"
```

**Ajouter des expressions locales** :
```python
# Pour le français québécois
"correct", "malade", "sick"

# Pour le français belge
"chouette", "formidable"

# Pour le français africain
"c'est fort", "oh là"
```

**Ajouter des expressions anglaises** :
```python
"awesome", "amazing", "great", "fantastic", "wonderful", "yay"
```

---

### 2. Seuil de Détection Joviale

**Fichier** : `Back/rag_core/llm.py` (ligne ~57)

```python
# Configuration actuelle : 2 marqueurs OU 2 exclamations
is_jovial = sum(1 for marker in jovial_markers if marker in user_text_combined.lower()) >= 2 or user_text_combined.count("!") >= 2
```

#### 🔧 Ajustements Possibles

**Plus sensible** (détecte plus facilement) :
```python
# 1 marqueur OU 1 exclamation suffit
is_jovial = sum(1 for marker in jovial_markers if marker in user_text_combined.lower()) >= 1 or user_text_combined.count("!") >= 1
```

**Moins sensible** (détecte moins facilement) :
```python
# 3 marqueurs OU 3 exclamations
is_jovial = sum(1 for marker in jovial_markers if marker in user_text_combined.lower()) >= 3 or user_text_combined.count("!") >= 3
```

**Basé sur un score** (recommandé pour production) :
```python
# Score pondéré : emojis valent 2 points, mots 1 point, ! vaut 0.5 point
emoji_count = sum(1 for marker in ["😊","😄","😃","🙂","👍","😁","🎉","✨"] if marker in user_text_combined)
word_count = sum(1 for marker in ["haha","lol","cool","super","génial","top","nickel","excellent"] if marker in user_text_combined.lower())
exclaim_count = user_text_combined.count("!")

jovial_score = (emoji_count * 2) + word_count + (exclaim_count * 0.5)
is_jovial = jovial_score >= 3  # Seuil à 3 points
```

---

### 3. Profondeur Historique

**Fichier** : `Back/rag_core/llm.py` (ligne ~56)

```python
# Analyse des 3 derniers messages utilisateur
user_text_combined = question + " " + " ".join([m.get("content","") for m in (history or [])[-3:] if m.get("role")=="user"])
```

#### 🔧 Ajustements

**Historique plus court** (réponse plus réactive) :
```python
# Analyse des 2 derniers messages
[...][-2:]
```

**Historique plus long** (adaptation plus progressive) :
```python
# Analyse des 5 derniers messages
[...][-5:]
```

**Historique pondéré** (messages récents comptent plus) :
```python
# Message actuel compte double
user_text_combined = (question * 2) + " " + " ".join([m.get("content","") for m in (history or [])[-3:] if m.get("role")=="user"])
```

---

### 4. Instructions IA par Mode

#### Mode Small Talk

**Fichier** : `Back/rag_core/llm.py` (lignes ~95-98)

**Actuel (Jovial)** :
```python
"L'utilisateur est jovial ! Adopte un ton enthousiaste et utilise des emojis pour rendre la conversation plus vivante et chaleureuse. 😊✨"
```

**Options de personnalisation** :

*Plus d'emojis* :
```python
"L'utilisateur est très jovial ! Utilise beaucoup d'emojis pour refléter son enthousiasme ! 😊🎉✨💪"
```

*Moins d'emojis* :
```python
"L'utilisateur est jovial. Reste enthousiaste mais utilise les emojis avec parcimonie (1-2 max). 😊"
```

*Très enthousiaste* :
```python
"L'utilisateur est SUPER enthousiaste ! 🎉 Partage son énergie avec beaucoup d'émojis et d'exclamations ! ✨💪"
```

---

#### Mode Normal

**Fichier** : `Back/rag_core/llm.py` (lignes ~113-114)

**Actuel (Jovial)** :
```python
"L'utilisateur est jovial ! Adopte un ton enthousiaste et utilise des emojis pour rendre tes réponses plus vivantes. 😊"
```

**Options** :

*Plus subtil* :
```python
"L'utilisateur semble de bonne humeur. Adopte un ton positif et ajoute un emoji si approprié. 😊"
```

*Plus professionnel* :
```python
"L'utilisateur est jovial, mais garde un ton professionnel tout en étant chaleureux."
```

---

### 5. Température du Modèle

La température affecte la créativité des réponses.

#### Mode Small Talk
**Actuel** : `0.65`

```python
payload = {..., "temperature": max(temperature, 0.65), ...}
```

**Options** :
- `0.5` : Plus cohérent, moins créatif
- `0.7` : Bon équilibre (recommandé pour small talk jovial)
- `0.8` : Plus créatif, plus varié
- `0.9` : Très créatif (peut être trop fantaisiste)

#### Mode Roleplay
**Actuel** : `0.9`

#### Mode Normal
**Actuel** : `0.6` (sans contexte) / `0.45` (avec contexte)

---

## 🎨 Profils Préconfigurés

### Profil "Enthousiaste"
```python
# Paramètres recommandés
jovial_threshold = 1  # Très sensible
emoji_instructions = "Utilise beaucoup d'emojis ! 🎉✨"
temperature_smalltalk = 0.75
history_depth = 5  # Mémoire longue
```

### Profil "Équilibré" (Par défaut)
```python
jovial_threshold = 2
emoji_instructions = "Utilise des emojis appropriés. 😊"
temperature_smalltalk = 0.65
history_depth = 3
```

### Profil "Professionnel"
```python
jovial_threshold = 3  # Moins sensible
emoji_instructions = "Utilise un emoji subtil si approprié."
temperature_smalltalk = 0.55
history_depth = 2
```

### Profil "Corporate"
```python
jovial_threshold = 5  # Très peu sensible
emoji_instructions = "Reste professionnel, évite les emojis sauf exceptionnellement."
temperature_smalltalk = 0.5
history_depth = 1
```

---

## 🧪 Mode Debug

Pour déboguer la détection, ajouter après la ligne 57 dans `llm.py` :

```python
# DEBUG : Afficher les détections
if True:  # Mettre False pour désactiver
    print(f"[JOVIAL DEBUG]")
    print(f"  Question: {question[:50]}...")
    print(f"  is_jovial: {is_jovial}")
    print(f"  is_casual: {is_casual}")
    print(f"  is_formal: {is_formal}")
    print(f"  Markers found: {sum(1 for m in jovial_markers if m in user_text_combined.lower())}")
    print(f"  Exclamations: {user_text_combined.count('!')}")
    print(f"  History depth: {len([m for m in (history or []) if m.get('role')=='user'])}")
```

**Résultat dans les logs** :
```
[JOVIAL DEBUG]
  Question: Salut ! Super app ! 😄 J'adore vraiment ça...
  is_jovial: True
  is_casual: True
  is_formal: False
  Markers found: 4
  Exclamations: 2
  History depth: 0
```

---

## 📊 Monitoring & Analytics

### Métriques à Suivre

```python
# Ajouter dans routes_ask.py ou autre fichier de logging

# Compteur de détections
jovial_count = 0
neutral_count = 0
formal_count = 0

# Dans la fonction ask()
if is_jovial:
    jovial_count += 1
    # Log analytics
    print(f"[ANALYTICS] Jovial detection: {jovial_count} total")
```

### Logs Structurés

```python
import json
import datetime

log_entry = {
    "timestamp": datetime.datetime.now().isoformat(),
    "user_tone": "jovial" if is_jovial else ("formal" if is_formal else "neutral"),
    "casual": is_casual,
    "emoji_count": sum(1 for m in ["😊","😄","😃"] if m in question),
    "exclamations": question.count("!"),
    "response_length": len(answer)
}

print(json.dumps(log_entry))
```

---

## 🔐 Configuration par Utilisateur

Pour permettre aux utilisateurs de choisir leur style préféré :

### Base de Données (suggestion)

```sql
-- Table user_preferences
CREATE TABLE user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    ai_style VARCHAR(50) DEFAULT 'balanced',  -- 'enthusiastic', 'balanced', 'professional', 'corporate'
    allow_emojis BOOLEAN DEFAULT true,
    formality_level INT DEFAULT 2  -- 1=casual, 2=balanced, 3=formal
);
```

### Code d'Intégration

```python
# Dans llm.py, modifier la fonction
def ask_mistral_with_context(
    question: str,
    context_text: str,
    history: List[Dict] = None,
    user_preferences: Dict = None,  # NOUVEAU
    **kwargs
):
    # Récupérer les préférences
    prefs = user_preferences or {}
    style = prefs.get("ai_style", "balanced")
    allow_emojis = prefs.get("allow_emojis", True)
    
    # Ajuster les seuils selon le style
    if style == "enthusiastic":
        jovial_threshold = 1
    elif style == "professional":
        jovial_threshold = 3
    elif style == "corporate":
        jovial_threshold = 5
        allow_emojis = False
    else:  # balanced
        jovial_threshold = 2
    
    # Appliquer dans la détection
    is_jovial = (sum(...) >= jovial_threshold) and allow_emojis
```

---

## 📝 Bonnes Pratiques

### ✅ Recommandations

1. **Commencer avec les paramètres par défaut**
2. **Tester avec de vrais utilisateurs** avant d'ajuster
3. **Monitorer les logs** pendant 1-2 semaines
4. **Ajuster progressivement** les seuils
5. **Respecter les préférences utilisateur**

### ❌ À Éviter

1. Seuil trop bas (faux positifs)
2. Trop d'emojis (devient agaçant)
3. Ignorer le contexte formel
4. Forcer un style unique pour tous

---

## 🆘 Dépannage

### Problème : Trop de faux positifs
**Solution** : Augmenter `jovial_threshold` de 2 à 3

### Problème : Pas assez de détections
**Solution** : Diminuer `jovial_threshold` de 2 à 1

### Problème : Emojis inappropriés
**Solution** : Améliorer les instructions système

### Problème : Transition trop brusque
**Solution** : Augmenter `history_depth` de 3 à 5

---

**Date** : 15 octobre 2025  
**Version** : 1.0  
**Maintenu par** : Équipe Auxilium

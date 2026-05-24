# 🧪 Tests - IA Conversationnelle Chaleureuse

## Exemples de comportements attendus

### Test 1 : Utilisateur Jovial 😊

**Entrée utilisateur** :
```
Salut ! Super contenu aujourd'hui ! 😄 J'adore ce que vous faites !! 🎉
```

**Détection** :
- `is_jovial = True` (3 emojis + 2 points d'exclamation détectés)
- `is_casual = True` (utilisation de "tu")
- `is_fr = True`

**Comportement de l'IA** :
- Ton : Enthousiaste et chaleureux
- Utilisation d'emojis : OUI ✨
- Exemple de réponse attendue :
  > "Merci beaucoup ! 😊 Je suis ravie que ça te plaise ! N'hésite pas si tu as des questions ! ✨"

---

### Test 2 : Utilisateur Neutre/Professionnel

**Entrée utilisateur** :
```
Bonjour, comment puis-je configurer les paramètres d'email ?
```

**Détection** :
- `is_jovial = False` (pas de marqueurs jovials)
- `is_casual = False`
- `is_formal = False`
- `is_fr = True`

**Comportement de l'IA** :
- Ton : Naturel et sympathique
- Utilisation d'emojis : Occasionnelle, subtile
- Exemple de réponse attendue :
  > "Bonjour ! Pour configurer les emails, rendez-vous dans les paramètres, section 'Email'. Vous pourrez y définir vos préférences de notification et de synchronisation."

---

### Test 3 : Utilisateur Formel

**Entrée utilisateur** :
```
Bonjour, pourriez-vous s'il vous plaît m'indiquer la procédure pour l'export de données ?
```

**Détection** :
- `is_jovial = False`
- `is_casual = False`
- `is_formal = True` (utilisation de "vous" et "s'il vous plaît")
- `is_fr = True`

**Comportement de l'IA** :
- Ton : Professionnel et chaleureux
- Utilisation d'emojis : NON
- Vouvoiement maintenu
- Exemple de réponse attendue :
  > "Bonjour, bien sûr ! Pour exporter vos données, accédez au menu Paramètres > Export. Vous pourrez sélectionner le format souhaité (CSV, JSON) et lancer l'export. N'hésitez pas si vous avez besoin d'aide."

---

### Test 4 : Small Talk Jovial

**Entrée utilisateur** :
```
Salut ! Comment vas-tu ? 😊
```

**Détection** :
- Mode : `smalltalk_mode = True`
- `is_jovial = True` (emoji détecté)
- `is_casual = True`

**Comportement de l'IA** :
- Ton : Enthousiaste et amical
- Réponse courte (1-2 phrases)
- Utilisation d'emojis : OUI
- Exemple de réponse attendue :
  > "Salut ! Je vais super bien, merci ! 😊 Et toi, comment ça va ?"

---

### Test 5 : Conversation Continue Joviale

**Historique** :
```
User: "Génial ce nouveau feature ! 😄"
AI: "Merci ! 😊 Je suis content que ça te plaise !"
User: "C'est vraiment top !!"
```

**Nouvelle question** :
```
Et comment je peux l'utiliser ?
```

**Détection** :
- `is_jovial = True` (analyse des 3 derniers messages utilisateur)
- Contexte jovial maintenu

**Comportement de l'IA** :
- Continue sur un ton enthousiaste
- Emojis appropriés
- Exemple de réponse attendue :
  > "C'est super simple ! 😊 Clique sur le bouton en haut à droite, et tu verras toutes les options disponibles. Tu vas adorer ! ✨"

---

## 📊 Matrice de Détection

| Critère | Seuil | Marqueurs |
|---------|-------|-----------|
| **Jovial** | 2+ marqueurs OU 2+ `!` | 😊 😄 😃 🙂 👍 😁 🎉 ✨, "haha", "lol", "cool", "super", "génial", "top", "nickel", "excellent" |
| **Casual** | Présence | "tu ", "toi " (FR) / "bro", "buddy", "mate", "dude" (EN) |
| **Formel** | Présence | "vous", "svp" (FR) / "please", "sir", "madam" (EN) |

---

## 🎯 Scénarios de Test Recommandés

### Test A : Transition Ton Jovial → Neutre
1. Message jovial avec emojis
2. Question technique neutre
3. Vérifier que l'IA s'adapte progressivement

### Test B : Transition Ton Formel → Casual
1. Message formel avec vouvoiement
2. Message plus décontracté avec tutoiement
3. Vérifier que l'IA suit la préférence de l'utilisateur

### Test C : Conversation Longue
1. Plusieurs échanges avec ton constant
2. Vérifier que la détection reste cohérente
3. Analyser l'utilisation appropriée des emojis

### Test D : Multilingue
1. Tester en français puis en anglais
2. Vérifier que la détection fonctionne dans les deux langues
3. Valider l'adaptation culturelle

---

## 🔍 Points de Validation

- ✅ L'IA détecte correctement le ton jovial
- ✅ Les emojis sont utilisés de manière appropriée
- ✅ Le ton reste professionnel quand nécessaire
- ✅ La précision des réponses factuelles n'est pas compromise
- ✅ L'adaptation se fait naturellement dans les conversations longues
- ✅ Le vouvoiement/tutoiement est respecté
- ✅ Les emojis ne sont pas excessifs ou inappropriés

---

## 🐛 Cas Limites à Tester

1. **Faux positifs jovials** : "Je veux une réduction !!!" (urgence, pas joie)
2. **Messages courts** : "Ok" (insuffisant pour détecter le ton)
3. **Emojis négatifs** : 😢 😞 (ne doivent pas déclencher le ton jovial)
4. **Questions techniques avec exclamations** : "Erreur 500 !!" (pas jovial)
5. **Historique mixte** : Alternance rapide de tons différents

---

## 📝 Notes pour l'Amélioration Future

1. **Emojis négatifs** : Ajouter une détection des emojis tristes/colère
2. **Contexte émotionnel** : Détecter frustration, urgence, satisfaction
3. **Personnalisation** : Permettre aux utilisateurs de choisir le style
4. **A/B Testing** : Mesurer la satisfaction avec/sans adaptation
5. **Seuils dynamiques** : Ajuster selon le feedback utilisateur

---

**Date de création** : 15 octobre 2025  
**Version** : 1.0  
**Statut** : Prêt pour tests

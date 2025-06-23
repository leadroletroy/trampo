#!/bin/bash

echo "🔍 Recherche des fichiers suivis mais ignorés..."

# Pour chaque fichier suivi par Git
for file in $(git ls-files); do
    # Vérifie s’il est maintenant ignoré
    if git check-ignore -q "$file"; then
        echo "❌ Supprime de l’index : $file"
        git rm --cached "$file"
    fi
done

echo "✅ Nettoyage terminé. Tu peux maintenant faire un commit."


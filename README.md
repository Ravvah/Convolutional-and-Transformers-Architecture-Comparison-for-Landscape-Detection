# 1. Cloner le repo
git clone https://github.com/TON-USERNAME/ton-repo.git
cd ton-repo

# 2. Installer uv (si pas déjà fait)
pip install uv
# ou (recommandé)
curl -Ls https://astral.sh/uv/install.sh | sh

# 3. Créer l'environnement + installer les dépendances
uv sync

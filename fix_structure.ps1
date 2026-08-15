# Run this from the project root: C:\Users\thode\OneDrive\recommendation-system
# It rebuilds the src/ package structure and moves each flattened file back
# into its correct subfolder.

# 1. Create the folder structure
New-Item -ItemType Directory -Force -Path src, src\data, src\models, src\ranking, src\evaluation, src\explainability | Out-Null

# 2. Create the __init__.py files that make these importable packages
New-Item -ItemType File -Force -Path `
    src\__init__.py, `
    src\data\__init__.py, `
    src\models\__init__.py, `
    src\ranking\__init__.py, `
    src\evaluation\__init__.py, `
    src\explainability\__init__.py | Out-Null

# 3. Move each file into its correct subfolder
Move-Item -Force loader.py           src\data\loader.py
Move-Item -Force content.py          src\models\content.py
Move-Item -Force semantic.py         src\models\semantic.py
Move-Item -Force collaborative.py    src\models\collaborative.py
Move-Item -Force neural_cf.py        src\models\neural_cf.py
Move-Item -Force popularity.py       src\models\popularity.py
Move-Item -Force random_baseline.py  src\models\random_baseline.py
Move-Item -Force candidate_gen.py    src\ranking\candidate_gen.py
Move-Item -Force hybrid.py           src\ranking\hybrid.py
Move-Item -Force diversity.py        src\ranking\diversity.py
Move-Item -Force metrics.py          src\evaluation\metrics.py
Move-Item -Force explain.py          src\explainability\explain.py
Move-Item -Force utils.py            src\utils.py

# 4. Move the saved model checkpoint into its own top-level models/ folder
#    (this is separate from src\models\ — one holds code, the other holds
#    the saved .pt weights file)
New-Item -ItemType Directory -Force -Path models | Out-Null
Move-Item -Force neural_cf.pt        models\neural_cf.pt

Write-Host "Done. Run 'python evaluate.py' now." -ForegroundColor Greenv
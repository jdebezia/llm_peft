# TP Text2SQL: Fine-tuning de LLM 

## Configuration de l'environnement
Ouvrez un terminal dans le JupyterLab puis exécutez la commande ci-dessous
```
source activate pytorch_p310 && cd SageMaker && sh env_configuration.sh 
```
## Dataset
Voici un aperçu du [dataset](https://huggingface.co/datasets/NumbersStation/NSText2SQL)

**ATTENTION**: Nous travaillons **uniquement** sur la base données **`atis`**

## Étapes:
1. Construisez votre dataset (`TP/01_process_datasets.ipynb`)
2. Évaluez le modèle non entrainé sur le dataset de test (`TP/02_zero_shot_inference.ipynb`)
3. Fine-tunez le modèle en utilisant les datasets de *train/validation* (libre à vous de les ajuster) (`TP/03_llm_peft.ipynb`)
4. Évaluez le modèle fine-tuné sur le dataset de *test* (avec les fonctions fournies) (`TP/04_fine_tuned_inference.ipynb`)
5. Générer les score d'évaluation (`TP/05_Evaluation.ipynb`)

## Objectif:
Avoir le meilleur score d'évalution donc n'hésitez pas à répéter les étapes **3.** & **4.** afin d'améliorer les performances.
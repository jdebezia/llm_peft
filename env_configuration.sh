pip install -q setuptools==70.0.0
pip install -q -r requirements.txt
huggingface-cli login --quiet --token $HG_TOKEN
echo "Installation terminée"
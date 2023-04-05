echo "#### RUNNING MYPY TESTS ####"
PYTHONDONTWRITEBYTECODE=1 mypy quantum_perceptron/

echo "#### RUNNING PYCODESTYLE TESTS ####"
pycodestyle quantum_perceptron/

echo "#### RUNNING PYTEST TESTS ####"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
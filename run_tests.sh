echo "#### RUNNING MYPY TESTS ####"
PYTHONDONTWRITEBYTECODE=1 mypy ./

echo "#### RUNNING PYCODESTYLE TESTS ####"
pycodestyle ./

echo "#### RUNNING PYTEST TESTS ####"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
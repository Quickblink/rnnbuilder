python3 run_tests.py
cd library
python3 -m build
python3 -m twine upload --repository pypi dist/*
rm -r dist
rm -r build
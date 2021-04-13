cd library
python3 -m build
python3 -m twine upload --repository testpypi dist/*
rm -r dist
rm -r build
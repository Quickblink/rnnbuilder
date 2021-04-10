pdoc --html library/src/rnnbuilder --force
rm -r docs
mv html/rnnbuilder docs
rm -r html
git add docs
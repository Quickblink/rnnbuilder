pdoc --html library/src/rnnbuilder --force
rm -r docs
mv html/rnnbuilder docs
rm -r html
cp -r imgsdoc docs/img
git add docs
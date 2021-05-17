pdoc --html library/src/rnnbuilder --force --config latex_math=True
rm -r docs
mv html/rnnbuilder docs
rm -r html
cp -r imgsdoc docs/img
git add docs
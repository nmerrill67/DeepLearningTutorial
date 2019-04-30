#/bin/bash

for file in *-eps-converted-to.pdf; do
  mv "$file" "${file%-eps-converted-to.pdf}.pdf" # Rename pdf
  rm "${file%-eps-converted-to.pdf}.eps" # remove stale eps file
done

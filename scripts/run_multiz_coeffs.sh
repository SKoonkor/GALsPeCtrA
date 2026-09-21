#!/usr/bin/env bash
# Build one PCA coefficient product per output redshift of a multi-snapshot run.
#
# process_lgalaxies.py deliberately has no default --sample and no default --output --
# a default output once silently overwrote a Millennium-I product with Millennium-II
# coefficients -- so every path is named here explicitly. The naming convention itself
# comes from galspectra.trees, queried once below, so this script does not restate it.
#
# The z = 0 product keeps its bare name, `lgalaxies_sed_coeffs_bc03.npz`. That is the
# convention `_unlabelled_output_simulation()` reads to decide what an unlabelled output
# filename means, and renaming it would break that check rather than tidy it.
#
# Usage (from the repository root):
#     scripts/run_multiz_coeffs.sh                # every sample on disk
#     scripts/run_multiz_coeffs.sh 0.00 1.04      # just these tags
#     OVERWRITE=1 scripts/run_multiz_coeffs.sh    # rebuild existing products

set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-./GALPCA_venv/bin/python}"
BACKEND="${BACKEND:-bc03}"

# No `mapfile`: the system bash on macOS is 3.2, which predates it. A plain word-split
# is safe here because a redshift tag is digits and a dot.
if [ "$#" -gt 0 ]; then
    TAGS="$*"
else
    TAGS=$("$PYTHON" -c "
from pathlib import Path
from galspectra.trees import get_tree
from galspectra.lgalaxies import discover_tags
lgal = Path('..').resolve() / 'L-GALAXIES' / 'LGalaxies2020_PublicRepository-master'
print(' '.join(discover_tags(get_tree('MR'), lgal)))")
fi

echo "Building coefficient product(s) for: ${TAGS}   (backend: ${BACKEND})"

for tag in ${TAGS}; do
    read -r sample output < <("$PYTHON" -c "
from pathlib import Path
from galspectra.trees import get_tree
tree = get_tree('MR')
lgal = Path('..').resolve() / 'L-GALAXIES' / 'LGalaxies2020_PublicRepository-master'
print(tree.sample_path(lgal, '$tag'), tree.coeffs_path(Path('data'), '$tag'))")

    echo
    echo "===== z${tag} ============================================================"
    if [ -f "$output" ] && [ -z "${OVERWRITE:-}" ]; then
        echo "  exists, skipping: $output   (set OVERWRITE=1 to rebuild)"
        continue
    fi

    # --redshift is left off on purpose: process_lgalaxies.py reads the sample's own
    # SnapNum and uses that snapshot's Planck redshift for the dust column density,
    # which is right in every case and cannot drift out of step with the sample.
    "$PYTHON" scripts/process_lgalaxies.py \
        --sample "$sample" \
        --output "$output" \
        --backend "$BACKEND" \
        ${OVERWRITE:+--overwrite}
done

echo
echo "Done."

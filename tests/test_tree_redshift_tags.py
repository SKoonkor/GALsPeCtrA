"""The redshift-tag half of `galspectra.trees`, and the snapshot mapping behind it."""

from __future__ import annotations

from pathlib import Path

import pytest

from galspectra.lgalaxies.snapshots import (Snapshot, format_tag, redshift_table,
                                            snapshot, snapshots)
from galspectra.trees import LABELS, get_tree

LGAL_ROOT = (Path(__file__).resolve().parents[2]
             / "L-GALAXIES" / "LGalaxies2020_PublicRepository-master")

requires_lgal = pytest.mark.skipif(
    not (LGAL_ROOT / "input" / "zlists" / "zlist_planck_MR.txt").exists(),
    reason="the L-GALAXIES tree (and its zlists) are not present",
)

ROOT = Path("/lgal")      # any root; these tests are about names, not the filesystem
DATA = Path("/data")


class TestTagNormalisation:
    """`None` and `"0.00"` are the same redshift and must name the same files."""

    @pytest.mark.parametrize("label", LABELS)
    def test_all_three_paths_agree_for_none_and_zero(self, label):
        tree = get_tree(label)
        assert tree.sample_path(ROOT) == tree.sample_path(ROOT, "0.00")
        assert tree.coeffs_path(DATA) == tree.coeffs_path(DATA, "0.00")
        assert tree.output_paths(ROOT) == tree.output_paths(ROOT, tag="0.00")

    def test_the_bare_coefficient_name_survives(self):
        """The convention `_unlabelled_output_simulation()` reads must not move."""
        assert get_tree("MR").coeffs_path(DATA).name == "lgalaxies_sed_coeffs_bc03.npz"
        assert get_tree("MR").coeffs_path(DATA, "0.00").name == \
            "lgalaxies_sed_coeffs_bc03.npz"

    def test_a_nonzero_tag_changes_every_path(self):
        tree = get_tree("MR")
        assert tree.sample_path(ROOT, "1.04") != tree.sample_path(ROOT)
        assert tree.coeffs_path(DATA, "1.04") != tree.coeffs_path(DATA)
        assert tree.output_paths(ROOT, tag="1.04") != tree.output_paths(ROOT)


class TestTagSpelling:
    """Both halves of the sample's `z<low>-<high>` range move together."""

    @pytest.mark.parametrize("label", LABELS)
    def test_sample_name_carries_the_tag_twice(self, label):
        name = get_tree(label).sample_path(ROOT, "2.44").name
        assert "z2.44-2.44" in name
        assert "0.00" not in name

    @pytest.mark.parametrize("label", LABELS)
    def test_binary_name_carries_it_once(self, label):
        tree = get_tree(label)
        first = tree.file_range.start
        assert f"z2.44_{first}" in tree.output_paths(ROOT, first, tag="2.44")[0].name

    def test_coefficient_name_gains_a_suffix(self):
        assert get_tree("MR").coeffs_path(DATA, "3.11").name == \
            "lgalaxies_sed_coeffs_bc03_z3.11.npz"

    @pytest.mark.parametrize("bad", ["1.0", "1.045", "z1.04", "", "abc", 1.04, 104])
    def test_a_malformed_tag_raises(self, bad):
        with pytest.raises(ValueError, match="two decimals"):
            get_tree("MR").sample_path(ROOT, bad)


class TestTagOfSample:
    """`tag_of_sample` is the inverse of `sample_path`, and refuses the other tree."""

    @pytest.mark.parametrize("label", LABELS)
    @pytest.mark.parametrize("tag", ["0.00", "0.26", "1.04", "5.03", "10.00"])
    def test_round_trip(self, label, tag):
        tree = get_tree(label)
        assert tree.tag_of_sample(tree.sample_path(ROOT, tag)) == tag

    def test_the_other_tree_is_not_claimed(self):
        """A Millennium-II sample must not yield a tag Millennium-I claims."""
        mr, mrii = get_tree("MR"), get_tree("MRII")
        assert mr.tag_of_sample(mrii.sample_path(ROOT, "1.04")) is None
        assert mrii.tag_of_sample(mr.sample_path(ROOT, "1.04")) is None

    @pytest.mark.parametrize("name", ["SFH_Bins", "notes.txt", "",
                                      "Planck_Mil-I_snapshots_default_test3_z1.04-2.07_All.npy"])
    def test_a_name_that_is_not_a_sample_returns_none(self, name):
        """A multi-redshift range (unequal z<first>-<last> halves) returns None."""
        assert get_tree("MR").tag_of_sample(name) is None


@requires_lgal
class TestSnapshotMapping:
    """The two redshift grids, read from the zlists rather than transcribed."""

    def test_z_zero_is_snapshot_58_not_the_last_one(self):
        """The rescaled box has five snapshots at negative redshift after z = 0."""
        z = redshift_table(LGAL_ROOT, "planck")
        assert len(z) == 64
        assert format_tag(z[58]) == "0.00"
        assert all(zi < 0 for zi in z[59:])

    def test_phot_redshift_is_the_wmap7_one(self):
        """`z_phot` must come from the WMAP7 grid — the tables `PhotPrefix` selects."""
        wmap7 = redshift_table(LGAL_ROOT, "wmap7")
        for snap in (19, 25, 30, 38, 45, 50):
            assert snapshot(LGAL_ROOT, snap).z_phot == pytest.approx(wmap7[snap])

    def test_phot_redshift_is_clamped_at_zero(self):
        """Snapshot 58 sits at WMAP7 z = -0.165; a spectrum cannot be blueshifted."""
        assert redshift_table(LGAL_ROOT, "wmap7")[58] < 0
        assert snapshot(LGAL_ROOT, 58).z_phot == 0.0
        assert all(snapshot(LGAL_ROOT, s).z_phot >= 0.0 for s in range(64))

    def test_the_two_grids_really_do_differ(self):
        """z_planck and z_phot differ by more than 0.1 at every mid-range snapshot."""
        for snap in (25, 30, 38, 45, 50):
            s = snapshot(LGAL_ROOT, snap)
            assert s.z_planck - s.z_phot > 0.1

    def test_tags_select_the_snapshots_the_model_wrote(self):
        wanted = ["0.00", "0.26", "0.51", "0.78", "1.04",
                  "1.48", "2.07", "2.44", "3.11", "5.03"]
        assert [s.snap for s in snapshots(LGAL_ROOT, tags=wanted)] == \
            [58, 50, 45, 41, 38, 34, 30, 28, 25, 19]

    def test_every_non_negative_snapshot_has_a_unique_tag(self):
        """Two snapshots sharing a tag would share an output filename."""
        tags = [format_tag(z) for z in redshift_table(LGAL_ROOT, "planck") if z >= 0]
        assert len(tags) == len(set(tags))

    def test_an_unknown_tag_names_the_alternatives(self):
        with pytest.raises(ValueError, match="Available tags"):
            snapshots(LGAL_ROOT, tags=["9.99"])

    def test_exactly_one_selector_is_required(self):
        with pytest.raises(ValueError, match="exactly one"):
            snapshots(LGAL_ROOT)
        with pytest.raises(ValueError, match="exactly one"):
            snapshots(LGAL_ROOT, tags=["0.00"], snaps=[58])

    def test_out_of_range_snapshot_raises(self):
        with pytest.raises(ValueError, match="outside"):
            snapshot(LGAL_ROOT, 99)

    def test_the_record_derives_its_own_tag(self):
        """`tag` is a property, so it cannot disagree with `z_planck`."""
        s = snapshot(LGAL_ROOT, 28)
        assert isinstance(s, Snapshot)
        assert s.tag == format_tag(s.z_planck) == "2.44"


class TestFileNumbers:
    """file_numbers() and output_paths() stay the same list, in the same order."""

    @pytest.mark.parametrize("label", LABELS)
    def test_none_means_the_whole_range(self, label):
        tree = get_tree(label)
        assert tree.file_numbers() == list(tree.file_range)

    @pytest.mark.parametrize("label", LABELS)
    def test_numbers_and_paths_correspond(self, label):
        """Same length, same order — the property the zip in notebook 09 relies on."""
        tree = get_tree(label)
        for file_nr in (None, tree.file_range.start, tree.file_range[-1]):
            nrs = tree.file_numbers(file_nr)
            paths = tree.output_paths(ROOT, file_nr)
            assert len(nrs) == len(paths)
            for n, path in zip(nrs, paths):
                assert path.name == tree.output_pattern.format(n=n)

    @pytest.mark.parametrize("label", LABELS)
    def test_a_single_number_gives_a_single_file(self, label):
        tree = get_tree(label)
        first = tree.file_range.start
        assert tree.file_numbers(first) == [first]
        assert len(tree.output_paths(ROOT, first)) == 1

    @pytest.mark.parametrize("label", LABELS)
    def test_out_of_range_raises_from_both(self, label):
        """The guard must survive being moved out of `output_paths`."""
        tree = get_tree(label)
        outside = tree.file_range.stop + 1
        with pytest.raises(ValueError, match="outside"):
            tree.file_numbers(outside)
        with pytest.raises(ValueError, match="outside"):
            tree.output_paths(ROOT, outside)

    def test_the_tag_does_not_disturb_the_ordering(self):
        tree = get_tree("MR")
        assert (len(tree.output_paths(ROOT, tag="2.44"))
                == len(tree.file_numbers()) == len(tree.output_paths(ROOT)))

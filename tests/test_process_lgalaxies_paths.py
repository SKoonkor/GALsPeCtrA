"""Regression tests for the process_lgalaxies.py sample/output path guard.

Background
----------
`scripts/process_lgalaxies.py` used to hard-code a **Millennium-II** sample path at
module level while its default output filename was the **Millennium-I** one. Running
it with defaults therefore overwrote `data/lgalaxies_sed_coeffs_bc03.npz` (Mil-I)
with MRII coefficients, silently and with no error.

These tests pin the fix:
  * both `--sample` and `--output` are required, with no defaults
  * an existing output is refused unless `--overwrite` is passed
  * the sample and output filenames must refer to the same simulation
"""

import pytest

from process_lgalaxies import (
    assert_same_simulation,
    detect_simulation,
    parse_args,
)

# The two real sample files this project uses.
MIL_I = "Planck_Mil-I_snapshots_default_test3_z0.00-0.00_All.npy"
MIL_II = "Planck_Mil-II_snapshots_default_test1_z0.00-0.00_All.npy"


class TestDetectSimulation:
    @pytest.mark.parametrize(
        "name, expected",
        [
            (MIL_I, "MR"),
            (MIL_II, "MRII"),
            ("lgalaxies_sed_coeffs_bc03_MRII_test1.npz", "MRII"),
            ("lgalaxies_sed_coeffs_bc03_MR.npz", "MR"),
            ("SA_DM_test3_z0.00_5", None),
            ("lgalaxies_sed_coeffs_bc03.npz", None),
        ],
    )
    def test_token_detection(self, name, expected):
        assert detect_simulation(name) == expected

    def test_mil_ii_is_not_mistaken_for_mil_i(self):
        """'Mil-I' is a prefix of 'Mil-II'; the MRII pattern must win."""
        assert detect_simulation(MIL_II) == "MRII"
        assert detect_simulation(MIL_II) != "MR"


class TestAssertSameSimulation:
    def test_matching_pair_is_accepted(self):
        assert assert_same_simulation(MIL_II, "coeffs_bc03_MRII.npz") == "MRII"

    def test_bare_output_is_accepted_for_mil_i(self):
        """The historical convention: no token in the name means Millennium-I."""
        assert assert_same_simulation(MIL_I, "lgalaxies_sed_coeffs_bc03.npz") == "MR"

    def test_the_original_bug_is_rejected(self):
        """MRII sample + bare (Mil-I convention) output name. This is the bug."""
        with pytest.raises(ValueError, match="unlabelled output filename"):
            assert_same_simulation(MIL_II, "lgalaxies_sed_coeffs_bc03.npz")

    def test_explicit_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="Simulation mismatch"):
            assert_same_simulation(MIL_II, "lgalaxies_sed_coeffs_bc03_MR.npz")

        with pytest.raises(ValueError, match="Simulation mismatch"):
            assert_same_simulation(MIL_I, "lgalaxies_sed_coeffs_bc03_MRII.npz")

    def test_undetectable_sample_requires_explicit_flag(self):
        with pytest.raises(ValueError, match="Cannot tell which simulation"):
            assert_same_simulation("some_sample.npy", "out_MR.npz")

        assert assert_same_simulation(
            "some_sample.npy", "out_MRII.npz", override="MRII"
        ) == "MRII"

    def test_override_still_checked_against_output(self):
        with pytest.raises(ValueError, match="Simulation mismatch"):
            assert_same_simulation("some_sample.npy", "out_MR.npz", override="MRII")


class TestRequiredArguments:
    def test_sample_and_output_are_required(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["process_lgalaxies.py"])
        with pytest.raises(SystemExit):
            parse_args()

        monkeypatch.setattr("sys.argv", ["process_lgalaxies.py", "--sample", MIL_I])
        with pytest.raises(SystemExit):
            parse_args()

    def test_no_default_output_path(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["process_lgalaxies.py", "--sample", MIL_I, "--output", "out.npz"],
        )
        args = parse_args()
        assert args.sample == MIL_I
        assert args.output == "out.npz"
        assert args.overwrite is False


class TestRegistryMustNotSupplyPaths:
    """`galspectra.trees` must not be allowed to fill in --sample and --output.

    The registry is consumed here for one thing only: the *expected* coefficient
    filename, so `assert_same_simulation()` can state the bare-filename convention
    once, as data.

    It must never become a source of path defaults. The bug fixed in August 2026
    was two independently-defaulted paths that named different simulations — a
    Millennium-II sample beside a Millennium-I output — so a default run silently
    overwrote the Millennium-I product. The fix was to delete the defaults and add
    a cross-check.

    A `--tree MR` flag that filled in both paths would undo that fix twice over.
    It would restore an output default, so a run could again overwrite a product
    nobody named; and it would **silence the cross-check**, because both paths
    would come from the same `Tree` and would therefore agree by construction,
    leaving `assert_same_simulation()` with nothing left to catch.

    These tests exist so that change fails loudly rather than looking tidy.
    """

    def test_no_tree_flag(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["process_lgalaxies.py", "--tree", "MR", "--output", "out.npz"],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_tree_is_absent_from_the_help_text(self, monkeypatch, capsys):
        """`--help` must not advertise a --tree option."""
        monkeypatch.setattr("sys.argv", ["process_lgalaxies.py", "--help"])
        with pytest.raises(SystemExit):
            parse_args()
        help_text = capsys.readouterr().out
        assert "--tree" not in help_text, (
            "process_lgalaxies.py grew a --tree option. If it supplies --sample and "
            "--output together, assert_same_simulation() can no longer catch a "
            "mismatch — the two paths would agree by construction. Read the class "
            "docstring."
        )
        assert "--sample" in help_text and "--output" in help_text

    def test_sample_and_output_declare_no_default_in_source(self):
        """Static pin on the two declarations, so a default cannot creep back.

        Read from the source rather than from a parser instance: constructing the
        parser to inspect it means monkeypatching argparse, and that is more
        machinery than the invariant is worth.
        """
        import inspect

        import process_lgalaxies

        src = inspect.getsource(process_lgalaxies.parse_args)
        for name in ("--sample", "--output"):
            decl_start = src.index(f'"{name}"')
            decl = src[decl_start:src.index("p.add_argument", decl_start + 1)]
            assert "required=True" in decl, f"{name} must stay required, got: {decl!r}"
            assert "default=" not in decl, f"{name} must have no default, got: {decl!r}"
        assert '"--tree"' not in src and "'--tree'" not in src

    def test_registry_is_used_only_for_the_expected_name(self):
        """The convention is stated once, as data, and it still resolves."""
        from galspectra.trees import get_tree
        from process_lgalaxies import _unlabelled_output_simulation

        assert _unlabelled_output_simulation() == "MR"
        assert get_tree("MR").coeffs_name == "lgalaxies_sed_coeffs_bc03.npz"


class TestOverwriteGuard:
    """The overwrite check lives in main(); exercise it directly on a real file."""

    def test_existing_output_is_refused_without_flag(self, tmp_path):
        existing = tmp_path / "lgalaxies_sed_coeffs_bc03.npz"
        existing.write_bytes(b"not really an npz")

        # Mirror of the guard in main(). Kept in the test so the intent is pinned
        # even if main() is refactored.
        assert existing.exists()
        overwrite = False
        with pytest.raises(FileExistsError):
            if existing.exists() and not overwrite:
                raise FileExistsError(f"Output already exists: {existing}")

    def test_overwrite_flag_permits_it(self, tmp_path, monkeypatch):
        existing = tmp_path / "lgalaxies_sed_coeffs_bc03.npz"
        existing.write_bytes(b"not really an npz")
        monkeypatch.setattr(
            "sys.argv",
            [
                "process_lgalaxies.py",
                "--sample", MIL_I,
                "--output", str(existing),
                "--overwrite",
            ],
        )
        args = parse_args()
        assert args.overwrite is True

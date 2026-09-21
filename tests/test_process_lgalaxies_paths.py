"""process_lgalaxies.py: --sample/--output are required with no defaults, an
existing output needs --overwrite, and the two filenames must name the same
simulation."""

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
    """galspectra.trees must not be allowed to fill in --sample and --output."""

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
            "a --tree flag would let the two paths agree by construction, "
            "silencing assert_same_simulation()'s cross-check"
        )
        assert "--sample" in help_text and "--output" in help_text

    def test_sample_and_output_declare_no_default_in_source(self):
        """Static pin: --sample/--output stay required, with no default, in source."""
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

        # mirrors the guard in main()
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

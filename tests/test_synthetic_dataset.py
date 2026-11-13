"""The synthetic example dataset (examples/synthetic_dataset) must build, import through every importer, and run
through a grid, facies, data checks and image imports. Guards the dataset and the importers against drifting apart."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parent.parent / "examples" / "synthetic_dataset"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EX / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    mk = _load("make_dataset")
    info = mk.main(out / "ds")
    return out / "ds", info


def test_files_exist_and_are_deterministic(dataset, tmp_path):
    root, info = dataset
    assert len(list((root / "4_wells" / "logs").glob("*.las"))) == 16
    mk = sys.modules["make_dataset"]
    again = mk.main(tmp_path / "again")
    assert again["wells"] == info["wells"]
    assert (root / "4_wells" / "core_5_wells_md_percent.csv").read_bytes() == (tmp_path / "again" / "4_wells" / "core_5_wells_md_percent.csv").read_bytes()
    assert (root / "README.md").exists() and "synthetic" in (root / "README.md").read_text().lower()


def test_truth_properties(dataset):
    _, info = dataset
    assert info["facies_rules_vs_truth"]["fraction_correct"] > 0.9
    assert info["wells"]["W-06"]["crosses_faults_inside_reservoir"] == ["F_Minor"]
    assert "Basement" not in info["wells"]["W-12"]["picks_md"] and "Res_B" not in info["wells"]["W-13"]["picks_md"]
    for iv in ("Res_A", "Res_B"):
        assert abs(info["image_slices"][iv]["sand_share_in_image"] - info["image_slices"][iv]["sand_share_interval"]) < 0.08


def test_every_file_imports_and_the_pipeline_runs(dataset):
    root, _ = dataset
    vf = _load("verify_with_cosma")
    pr = vf.run(root)
    assert len(pr.wells) == 16 and sorted(pr.faults) == ["F_East", "F_Main", "F_Minor"]
    assert [i.name for i in pr.chart.intervals] == ["Seal", "Res_A", "Res_B"]
    # the fault dips recovered from the sticks agree with the truth
    assert pr.faults["F_Main"].fit().dip_deg == pytest.approx(86.0, abs=1.5)
    assert pr.faults["F_East"].fit().dip_deg == pytest.approx(88.0, abs=2.5)       # imported from feet
    vf.pipeline(pr, root)
    assert pr.fault_age_order()[0] == "F_Main"

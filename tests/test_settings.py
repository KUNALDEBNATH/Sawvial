import os

from sawvial import settings


def test_env_file_is_loaded_but_real_environment_wins(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nSAWVIAL_TEST_A=from_file\nSAWVIAL_TEST_B=\"quoted\"\nSAWVIAL_TEST_C=\nOTHER=ignored\nbroken line\n")
    monkeypatch.setenv("SAWVIAL_TEST_A", "real")
    for k in ("SAWVIAL_TEST_B", "SAWVIAL_TEST_C", "OTHER"):
        monkeypatch.delenv(k, raising=False)
    n = settings.load_env_file(f)
    assert os.environ["SAWVIAL_TEST_A"] == "real"            # not overridden
    assert os.environ["SAWVIAL_TEST_B"] == "quoted" and n == 1
    assert "SAWVIAL_TEST_C" not in os.environ and "OTHER" not in os.environ
    monkeypatch.delenv("SAWVIAL_TEST_B", raising=False)


def test_missing_env_file_is_fine(tmp_path):
    assert settings.load_env_file(tmp_path / "nope.env") == 0

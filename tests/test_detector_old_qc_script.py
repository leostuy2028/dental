# Runs the old self-test script detector/test_prepare_qc.py.
# That script plants a defect in a tiny fake dataset (duplicate tooth, too many
# boxes, etc.) and checks the data-hygiene gate in prepare_data.py rejects it.
# It makes its own temp folder and deletes it afterwards.

import test_prepare_qc


def test_old_qc_script_passes(capsys):
    result = test_prepare_qc.main()
    printed = capsys.readouterr().out
    assert result == 0
    assert "ALL PASS" in printed
    assert "FAIL]" not in printed

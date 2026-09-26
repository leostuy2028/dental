# Tests

These tests lock in what the code does **today**, so a rewrite can be checked
against them. If the rewrite changes a result, a test goes red.

## How to run

```
python -m pytest                                        # run all tests
python -m pytest tests/test_parsing.py                  # run one file
python -m coverage run -m pytest ; python -m coverage report   # coverage per file
```

## Style (keep it simple)

Write tests the way a high school student would: plain and easy to read.

- One test file per source file: `clients/parsing.py` -> `tests/test_parsing.py`.
  If two source files share a name, add the folder: `tests/test_detector_validate.py`.
- Put a short comment at the top of the file saying what the source file does.
- Test names say what they check: `test_bare_letter_with_punctuation`.
- A test is: set something up, call the function, `assert` the answer.
- Plain functions only. No test classes.
- Fixtures: only pytest's built-in ones (`tmp_path` for temp files, `monkeypatch`
  to swap something out, `capsys` to read printed output). No custom fixture magic.
- To fake a model API, write a tiny fake class or function by hand and swap it in
  with `monkeypatch`. No `unittest.mock`, no mocking libraries.
- Make tiny test data inside the test (a 3-row DataFrame, a 10x10 image, a small
  JSON file in `tmp_path`). Do not depend on the big files in `data/` or `results/`.
- Comments only where something is surprising.

## Rules

- **Never change the source code to make a test pass.** The tests describe the
  code as it is.
- **Expected values must come from the real code**: read it, run it, and assert
  what it actually returns. Never guess an answer.
- If the code does something that looks wrong, still test the current behavior,
  add a comment starting with `# CURRENT BEHAVIOR (looks like a bug):`, and list it
  in `BUGS_FOUND.md`.
- No test may call a real model API. `conftest.py` blocks the internet and uses fake
  API keys, so an accidental real call fails instead of costing money.

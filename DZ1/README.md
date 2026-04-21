# DZ1

Homework for `01-intro-and-kinematics`.

## Contents

- `homework.ipynb` - notebook with task statements and visualizations
- `solutions/` - final solutions for all 3 problems
- `tests/` - local tests
- `assets/`, `lib/` - support files required by the notebook and tests

## Setup

Clone the repository and open the `DZ1` directory:

```powershell
git clone https://github.com/Sergo20025/AI-for-Robotics.git
cd AI-for-Robotics\DZ1
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Test solutions

Run the three task test suites:

```powershell
pytest tests/test_so101_ik.py -q
pytest tests/test_broom_racing.py -q
pytest tests/test_beads.py -q
```

Expected local results:

- `test_so101_ik.py`: `60 passed, 2 skipped`
- `test_broom_racing.py`: `8 passed, 8 skipped`
- `test_beads.py`: `6 passed, 6 skipped`

`skipped` tests are normal here because some reference and hidden checks are not available in the public repository.

## Notebook

If you want to inspect the tasks visually:

```powershell
jupyter lab
```

Then open `homework.ipynb`.

Note: `pyrender` is included in `requirements.txt` because it is required for some notebook visualizations.

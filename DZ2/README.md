# DZ2

Homework for `02-dynamics`.

## Contents

- `homework.ipynb` - notebook with task statements and demo commands
- `solutions/` - final solutions for the coding tasks
- `tests/` - local test suite
- `lib/`, `scripts/`, `container/` - support code and demo scripts

## Setup

Clone the repository and open the `DZ2` directory:

```powershell
git clone https://github.com/Sergo20025/AI-for-Robotics.git
cd AI-for-Robotics\DZ2
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

## Test Solutions

Run the local test suite:

```powershell
pytest tests -q
```

Expected local result:

- `55 passed`

Warnings may appear during the run, but the key condition is that there are no `failed` tests.

## Notebook And Demos

If you want to inspect the tasks visually:

```powershell
jupyter lab
```

Then open `homework.ipynb`.

Optional demo commands used in the notebook:

```powershell
python scripts/kin_energy.py --solver euler
python scripts/kin_energy.py --solver rk4
python scripts/bs_joint.py
python scripts/penalty.py
```

Note: `vpython` is included in `requirements.txt` for the visualization scripts, and `python-fcl` is needed for the collision-based parts.

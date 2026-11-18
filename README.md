# Cross-Entropy versus Robust Cross-Entropy

Run these commands from the project directory. The supplied results were generated with the Python and package versions recorded in `results/protocol.json` and `requirements.txt`.

```text
python -m pip install -r requirements.txt
python -B checks.py
python -B experiment.py
python -B report_results.py
tectonic main.tex
```

An existing TeX installation can also compile the report with `latexmk -pdf main.tex`.

- `losses.py` implements the three binary objectives and their logit derivatives. `models.py` implements the data generators and models.
- `experiment.py` runs clean validation tuning, records the selected settings, then runs the matched corruption study. Its random streams are deterministic. Training times depend on the machine and its load.
- `results/` contains all successful tuning and evaluation runs, learning curves, the run configuration, the failure log, and the fitted parameters and test points used for the boundary plots.
- `report_results.py` regenerates the vector PDF figures and LaTeX tables from the saved results. Rerunning the experiment overwrites the results, and rerunning this script overwrites the generated figures and tables.

The initial rate grid ended at 0.60. During clean validation selection, semicircle label smoothing selected that endpoint. The final grid adds 0.90, 1.20 and 1.80 for all methods in both simulations.

# 03 Parameter Sensitivity

Paper subsection: `Parameter Sensitivity`.

Run a quick smoke test:

```bash
python run_parameter_sensitivity.py --trials 2 --max-steps 70
```

Run the table-level experiment:

```bash
python run_parameter_sensitivity.py --trials 20
```

Outputs:

- `outputs/sensitivity_trials.csv`
- `outputs/sensitivity_summary.csv`
- first-trial trajectory PDFs for each sensitivity group.

The copied `sensitivity_summary.csv` and `sensitivity_trials.csv` in this
directory are the existing generated records from the working tree and are kept
as reference data.


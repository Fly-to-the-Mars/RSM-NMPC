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

`paper_sensitivity_table.csv` contains the exact manuscript table values. The
copied `sensitivity_summary.csv` and `sensitivity_trials.csv` are generated
reference records from the lightweight simulator; new runs write fresh results
to `outputs/` and may differ slightly with solver timing and random seeds.

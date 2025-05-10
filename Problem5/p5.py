# Problem5/p5.py
try:
    from common import experiment
except ModuleNotFoundError:
    # add project root to PYTHONPATH and retry
    import pathlib, sys
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from common import experiment
    print("Path fixed automatically. "
          "Next time, run with `python -m Problem5.p5` from the project root "
          "to avoid this message.", file=sys.stderr)

# run the experiment
experiment.run(include_heuristic=True, csv_name="p5_results.csv")

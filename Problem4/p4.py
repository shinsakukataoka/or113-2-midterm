# Problem4/p4.py
try:
    from common import experiment
except ModuleNotFoundError:
    import pathlib, sys
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from common import experiment
    print(
        "Path fixed automatically. "
        "Next time, run with `python -m Problem4.p4` from the project root "
        "to avoid this message.",
        file=sys.stderr,
    )

# LP + naïve only
experiment.run(include_heuristic=False, csv_name="p4_results.csv")

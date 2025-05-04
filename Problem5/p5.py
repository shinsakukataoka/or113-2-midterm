# run_experiment.py
import pandas as pd
import numpy as np
import os
import time
from solver_lp import solve_lp
from solver_proposed import solve_proposed
from solver_naive import solve_naive

# --- Configuration ---
INSTANCE_FOLDER = "instances"
OUTPUT_FOLDER   = "output"
RESULTS_FILE    = os.path.join(OUTPUT_FOLDER, "p5_results.csv")
NUM_SCENARIOS = 7
NUM_INSTANCES_PER_SCENARIO = 30

# --- Helper Function to Load Instance Data ---
def load_instance_data(scenario_id, instance_id):
    """Loads data for a specific instance from its .npz file."""
    file_name = f"scenario_{scenario_id}_instance_{instance_id}.npz"
    file_path = os.path.join(INSTANCE_FOLDER, file_name)
    try:
        with np.load(file_path) as data:
            # Convert back to a dictionary, handling 0-d arrays
            instance_data = {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data}
        return instance_data
    except FileNotFoundError:
        print(f"Error: Instance file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading instance {file_path}: {e}")
        return None

# --- Main Experiment Loop ---
def run_experiment():
    results_list = []
    print("Starting experiment...")

    for s_id in range(1, NUM_SCENARIOS + 1):
        print(f"--- Running Scenario {s_id} ---")
        for i_id in range(1, NUM_INSTANCES_PER_SCENARIO + 1):
            print(f"  Running Instance {i_id}...")
            instance_data = load_instance_data(s_id, i_id)
            if instance_data is None:
                print(f"    Skipping instance ({s_id}, {i_id}) due to loading error.")
                continue

            # Run LP Relaxation
            start_time = time.time()
            lp_obj, lp_status = solve_lp(instance_data) # Expecting objective value and status
            lp_time = time.time() - start_time
            # Handle cases where LP might be infeasible or unbounded
            if lp_status not in ["Optimal", "Feasible"]: # Adjust based on solver's status strings
                 print(f"    Warning: LP solver status for ({s_id}, {i_id}): {lp_status}. Setting obj to infinity.")
                 lp_obj = float('inf') # Or handle appropriately

            # Run Proposed Heuristic
            start_time = time.time()
            proposed_obj = solve_proposed(instance_data) # Expecting objective value
            proposed_time = time.time() - start_time

            # Run Naive Heuristic
            start_time = time.time()
            naive_obj = solve_naive(instance_data) # Expecting objective value
            naive_time = time.time() - start_time

            # Calculate Optimality Gaps (relative to LP relaxation)
            # Handle cases where LP obj is 0, negative, or infinite
            lp_gap = 0.0 # Gap of LP relative to itself is 0
            if lp_obj > 0 and lp_obj != float('inf') and proposed_obj != float('inf'):
                proposed_gap = ((proposed_obj - lp_obj) / lp_obj) * 100.0
            else:
                proposed_gap = float('nan') # Not calculable

            if lp_obj > 0 and lp_obj != float('inf') and naive_obj != float('inf'):
                 naive_gap = ((naive_obj - lp_obj) / lp_obj) * 100.0
            else:
                 naive_gap = float('nan') # Not calculable


            # Append results for this instance
            results_list.append({
                "Scenario ID": s_id,
                "Instance ID": i_id,
                "LP Relaxation": lp_obj if lp_obj != float('inf') else np.nan, # Store nan for non-optimal LPs maybe?
                "Proposed Algorithm": proposed_obj if proposed_obj != float('inf') else np.nan,
                "Naive Heuristic": naive_obj if naive_obj != float('inf') else np.nan,
                "LP Time": lp_time,
                "Proposed Time": proposed_time,
                "Naive Time": naive_time,
                "LP Gap": lp_gap, # This column seems redundant, maybe it was meant for Naive Gap? Using Proposed Gap here.
                "Proposed Gap": proposed_gap, # Renaming column 10 as "Proposed Gap"
                "Naive Gap": naive_gap # Adding Naive Gap as an extra column
            })

    print("Experiment finished. Compiling results...")

    # Create DataFrame and save to CSV
    if not results_list:
        print("No results generated.")
        return

    # Define column order as per instructions (slightly modified for clarity)
    columns_ordered = [
        "Scenario ID", "Instance ID",
        "LP Relaxation", "Proposed Algorithm", "Naive Heuristic",
        "LP Time", "Proposed Time", "Naive Time",
        "LP Gap", "Proposed Gap" # Using the original names from the prompt
        # If you want Naive Gap included, add it here and adjust the prompt's description
    ]
     # Adjusting based on the prompt's exact request (10 columns)
     # Prompt asks for: (1) scenario ID, (2) instance ID, (3)–(5) obj vals (LP, proposed, naive),
     # (6)–(8) comp times (LP, proposed, naive), and (9)–(10) two optimality gaps in percentages.
     # Let's assume the two gaps are Proposed vs LP and Naive vs LP.
    final_results_list = []
    for res in results_list:
         final_results_list.append({
             "Scenario ID": res["Scenario ID"],
             "Instance ID": res["Instance ID"],
             "LP Relaxation": res["LP Relaxation"],
             "Proposed Algorithm": res["Proposed Algorithm"],
             "Naive Heuristic": res["Naive Heuristic"],
             "LP Time": res["LP Time"],
             "Proposed Time": res["Proposed Time"],
             "Naive Time": res["Naive Time"],
             "Proposed Gap (%)": res["Proposed Gap"], # Gap 1
             "Naive Gap (%)": res["Naive Gap"]       # Gap 2
         })


    df_results = pd.DataFrame(final_results_list) # Use the list with gaps calculated

    # Reorder columns to match the required output structure more closely
    final_columns = [
        "Scenario ID", "Instance ID",
        "LP Relaxation", "Proposed Algorithm", "Naive Heuristic", # Cols 3-5
        "LP Time", "Proposed Time", "Naive Time", # Cols 6-8
        "Proposed Gap (%)", "Naive Gap (%)" # Cols 9-10 (Assuming these are the two gaps)
    ]
    df_results = df_results[final_columns]


    try:
        df_results.to_csv(RESULTS_FILE, index=False, float_format='%.4f')
        print(f"Results saved to '{RESULTS_FILE}'")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
    # Make sure you have PuLP installed: pip install pulp
    try:
        import pulp
    except ImportError:
        print("Error: PuLP library not found. Please install it using 'pip install pulp'")
        exit()

    run_experiment()
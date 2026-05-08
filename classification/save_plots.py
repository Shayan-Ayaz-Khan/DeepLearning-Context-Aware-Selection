"""
Post-processing script to automatically generate visualization plots from results.json
Usage: python save_plots.py <results_json_path>
"""
import sys
import json
import os
from pathlib import Path

# Add repo root to path to import visualise_results
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from visualise_results import generate_all_plots


def main():
    if len(sys.argv) < 2:
        print("Usage: python save_plots.py <results_json_path>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    if not os.path.isfile(results_path):
        print(f"Error: {results_path} not found")
        sys.exit(1)
    
    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # If multiple results, use the latest
    if isinstance(results, list):
        results = results[-1]
    
    # Get run directory
    run_dir = os.path.dirname(results_path)
    plot_dir = os.path.join(run_dir, "plots")
    
    # Stream config from results
    stream_config = {
        "stream_name": results.get("stream_name", "stream"),
        "corruption_boundaries": results.get("metrics_log", {}).get("corruption_boundaries", []),
    }
    
    print(f"Generating plots from: {results_path}")
    print(f"Saving plots to: {plot_dir}")
    
    try:
        figs = generate_all_plots(results, stream_config, output_dir=plot_dir)
        print(f"Successfully created {len(figs)} plots:")
        for name in figs.keys():
            print(f"  - {name}.png")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

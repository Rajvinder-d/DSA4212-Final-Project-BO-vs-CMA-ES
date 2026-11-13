"""V4-matched V5: print the plan, or use --run to execute a separate collection."""
from run_v5 import main


if __name__ == '__main__':
    raise SystemExit(main('configs/followup_v5/boundary_comparison_v4_aligned.json'))

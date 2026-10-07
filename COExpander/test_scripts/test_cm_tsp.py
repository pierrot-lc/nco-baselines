from pathlib import Path

from co_expander import (
    COExpanderCMModel,
    COExpanderDecoder,
    COExpanderEnv,
    COExpanderTSPSolver,
    TSPGNNEncoder,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--data", "-d", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--two-opt", action="store_true")
    parser.add_argument("--determinate-steps", type=int, default=3)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--sampling-num", type=int, default=4)
    parser.add_argument("--sparse-factor", type=int, default=-1, help="Should be 50 for TPS-500 and TSP-10k, 100 for TSP-1k, -1 for TSP-100 and TSP-50")
    args = parser.parse_args()

    solver = COExpanderTSPSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="TSP", sparse_factor=args.sparse_factor, device="cuda"
            ),
            encoder=TSPGNNEncoder(sparse=args.sparse_factor > 0),
            decoder=COExpanderDecoder(decode_kwargs={"use_2opt": args.two_opt}),
            weight_path=args.checkpoint,
            inference_steps=args.inference_steps,
            determinate_steps=args.determinate_steps,
        )
    )
    solver.from_npz(args.data)
    solver.solve(sampling_num=args.sampling_num, show_time=True)
    costs_avg, ref_costs_avg, gap_avg, gap_std = solver.evaluate(calculate_gap=True)
    print(f"gap: {gap_avg:.2f}")

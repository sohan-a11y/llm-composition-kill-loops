"""
CLI evaluation runner for Project C168: DNA Topoisomerase-II Strand-Passage Attention Unknotting.
"""

import sys
import argparse
import json
try:
    from projects.c168_topoisomerase_attention.evaluate import evaluate_c168, run_evaluation
except ImportError:
    from evaluate import evaluate_c168, run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate C168 Topoisomerase-II Attention")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--d_model", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--delta_knot", type=float, default=1.5, help="Topological crossing distance threshold")
    parser.add_argument("--lambda_strand", type=float, default=40.0, help="Strand-passage cleavage penalty")
    parser.add_argument("--seq_len", type=int, default=6, help="Sequence length")
    parser.add_argument("--n_train", type=int, default=400, help="Training samples")
    parser.add_argument("--n_test", type=int, default=300, help="Test samples")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi_seed", action="store_true", help="Run multi-seed evaluation")
    args = parser.parse_args()

    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_classes": args.n_classes,
        "delta_knot": args.delta_knot,
        "lambda_strand": args.lambda_strand,
        "seq_len": args.seq_len,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "lr": args.lr
    }

    if args.multi_seed:
        results = run_evaluation(config, seeds=[43, 138])
        print("\n================ MULTI-SEED EVALUATION SUMMARY ================")
        print(f"Topoisomerase Unknotting Acc: {results['mean_acc']:.4f} +/- {results['std_acc']:.4f}")
        print(f"Sanity Check (0 knots):       {results['sanity_acc']:.4f} (Threshold: >= 0.85)")
        print(f"Lethal Kill Arm (Anti-Topo):   {results['mean_kill_anti']:.4f} (Chance: {results['chance']:.4f})")
        print(f"Passed Sanity: {results['passed_sanity']} | Passed Kill-Control: {results['passed_kill']}")
        print("===============================================================\n")
    else:
        results = evaluate_c168(config, seed=args.seed)
        print("\n================ SINGLE-SEED EVALUATION SUMMARY ================")
        print(f"Sanity Check (0 knots):       {results['acc_sanity']:.4f} (Threshold: >= 0.85)")
        print(f"Topoisomerase Unknotting Acc: {results['acc_topo']:.4f}")
        print(f"Standard Attention Acc:       {results['acc_std']:.4f}")
        print(f"Lethal Control 1 (Anti-Topo): {results['acc_anti']:.4f} (Chance: {results['chance_baseline']:.4f})")
        print(f"Lethal Control 2 (Scrambled): {results['acc_scram']:.4f} (Chance: {results['chance_baseline']:.4f})")
        print(f"Passed Sanity: {results['passed_sanity']} | Passed Kill-Control: {results['passed_kill']}")
        print("================================================================\n")

if __name__ == "__main__":
    main()

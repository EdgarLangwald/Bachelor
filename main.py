from codebase.model import Model
from codebase.utils import load_dataset
from codebase.evaluate import evaluate

if __name__ == '__main__':
    alphas = [0, 0.25, 0.5, 0.75, 1]
    dataset = load_dataset("test_set/chunk_0.pkl")

    all_results = {}
    for alpha in alphas:
        model_path = f"alpha_{alpha}.pt"
        print(f"Evaluating {model_path}...")
        model = Model.load(model_path, "cuda")
        results = evaluate(model, dataset, num_samples=100, device="cuda", threshold=0.5, seed=42)
        all_results[alpha] = results
        print(f"  Done: correlation={results['correlation']:.4f}, f1={results['f1']:.4f}")

    with open("evaluation_results.md", "w") as f:
        f.write("# Model Evaluation Results\n\n")
        f.write("| Alpha | Correlation | Binary Corr | Precision | Recall | Accuracy | F1 | Tokens Ratio |\n")
        f.write("|-------|-------------|-------------|-----------|--------|----------|-----|-------------|\n")
        for alpha, r in all_results.items():
            f.write(f"| {alpha} | {r['correlation']:.4f} | {r['binary_correlation']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['num_tokens_ratio']:.4f} |\n")

    print("\nResults saved to evaluation_results.md")

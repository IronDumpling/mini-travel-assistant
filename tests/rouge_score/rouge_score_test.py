import os
import re
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

# Directory containing the test files
DATA_DIR = "rouge_score_samples"

# Regex to identify file numbers
pattern = re.compile(r"gpt_test_(\d+)\.txt")

# Initialize storage for scores
scores = {
    "deepseek": {"rouge1": [], "rouge2": [], "rougeL": []},
    "agent": {"rouge1": [], "rouge2": [], "rougeL": []},
    "index": [],
}

# Load all GPT baseline files and extract their <num>
baseline_files = sorted(f for f in os.listdir(DATA_DIR) if pattern.match(f))

# Create a ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for file in baseline_files:
    match = pattern.match(file)
    if not match:
        continue

    file_num = match.group(1)
    gpt_path = os.path.join(DATA_DIR, f"gpt_test_{file_num}.txt")
    deepseek_path = os.path.join(DATA_DIR, f"deepseek_test_{file_num}.txt")
    agent_path = os.path.join(DATA_DIR, f"agent_test_{file_num}.txt")

    # Check if comparison files exist
    if not (os.path.exists(deepseek_path) and os.path.exists(agent_path)):
        continue

    with open(gpt_path, 'r', encoding='utf-8') as f:
        gpt_text = f.read().strip()

    with open(deepseek_path, 'r', encoding='utf-8') as f:
        deepseek_text = f.read().strip()

    with open(agent_path, 'r', encoding='utf-8') as f:
        agent_text = f.read().strip()

    # Compute scores
    deepseek_scores = scorer.score(gpt_text, deepseek_text)
    agent_scores = scorer.score(gpt_text, agent_text)

    for metric in ['rouge1', 'rouge2', 'rougeL']:
        scores["deepseek"][metric].append(deepseek_scores[metric].fmeasure)
        scores["agent"][metric].append(agent_scores[metric].fmeasure)

    scores["index"].append(int(file_num))

# Sort results by index
sorted_indices = sorted(range(len(scores["index"])), key=lambda i: scores["index"][i])
for model in ["deepseek", "agent"]:
    for metric in ["rouge1", "rouge2", "rougeL"]:
        scores[model][metric] = [scores[model][metric][i] for i in sorted_indices]
scores["index"] = [scores["index"][i] for i in sorted_indices]

# Plotting
metrics = ['rouge1', 'rouge2', 'rougeL']
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(scores["index"], scores["deepseek"][metric], label="DeepSeek", marker='o')
    plt.plot(scores["index"], scores["agent"][metric], label="Agent", marker='x')
    plt.title(f"{metric.upper()} F1 Score Comparison")
    plt.xlabel("Test Number")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison_plot.png")
    plt.show()
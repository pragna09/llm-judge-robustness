# Save the main experiment script
experiment_code = '''
import json
import urllib.request
import random
import re
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# ============================================
# CONFIGURATION
# ============================================

LLAMA_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MISTRAL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
N_EXAMPLES = 100
RANDOM_SEED = 42

# ============================================
# DATA LOADING
# ============================================

def load_data():
    """Download and load AlpacaEval dataset."""
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
    urllib.request.urlretrieve(url, "data/alpaca_eval.json")
    
    with open("data/alpaca_eval.json", "r") as f:
        data = json.load(f)
    
    return data[:N_EXAMPLES]

# ============================================
# MODEL LOADING
# ============================================

def load_llama():
    """Load Llama 3.1 8B with rope_scaling fix."""
    config = AutoConfig.from_pretrained(LLAMA_NAME)
    if config.rope_scaling and "rope_type" in config.rope_scaling:
        config.rope_scaling["type"] = config.rope_scaling["rope_type"]
    
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

def load_mistral():
    """Load Mistral 7B."""
    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

# ============================================
# JUDGING
# ============================================

def judge_response(model, tokenizer, instruction, response, max_new_tokens=10):
    """Judge a single response. Returns integer score 1-5."""
    prompt = f\"\"\"You are an expert evaluator. Rate the following response to the instruction on a scale of 1-5, where:
1 = Completely incorrect or unhelpful
2 = Mostly incorrect, some relevant content
3 = Partially correct, significant issues
4 = Mostly correct, minor issues
5 = Fully correct and helpful

Respond with ONLY a single digit (1, 2, 3, 4, or 5). No explanation.

Instruction: {instruction}

Response: {response}

Rating:\"\"\"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True
    )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    match = re.search(r'\\b([1-5])\\b', generated.strip())
    return int(match.group(1)) if match else None

# ============================================
# PERTURBATIONS
# ============================================

def add_typos(text, typo_rate=0.05, seed=None):
    """Randomly swap adjacent letters in words."""
    if seed:
        random.seed(seed)
    words = text.split()
    for i in range(len(words)):
        if random.random() < typo_rate and len(words[i]) > 3:
            word = list(words[i])
            idx = random.randint(0, len(word)-2)
            word[idx], word[idx+1] = word[idx+1], word[idx]
            words[i] = ''.join(word)
    return ' '.join(words)

def add_filler(text, seed=None):
    """Add a filler phrase at the beginning."""
    fillers = [
        "Let me think about this carefully. ",
        "To answer your question, ",
        "I believe the answer is as follows. ",
        "This is an interesting question. "
    ]
    if seed:
        random.seed(seed)
    return random.choice(fillers) + text

def change_formatting(text):
    """Convert to bullet points."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) <= 1:
        return text
    return '\\n'.join([f"- {s}." for s in sentences])

def shuffle_sentences(text, seed=None):
    """Shuffle sentence order."""
    if seed:
        random.seed(seed)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) <= 1:
        return text
    random.shuffle(sentences)
    return '. '.join(sentences) + '.'

# ============================================
# MAIN EXPERIMENT
# ============================================

def run_experiment():
    """Run full experiment and save results."""
    print("Loading data...")
    data = load_data()
    
    print("Loading models...")
    llama_model, llama_tokenizer = load_llama()
    print("Llama loaded!")
    mistral_model, mistral_tokenizer = load_mistral()
    print("Mistral loaded!")
    
    results = []
    
    for i, item in enumerate(data):
        instruction = item["instruction"]
        response = item["output"]
        
        # Original scores
        llama_orig = judge_response(llama_model, llama_tokenizer, instruction, response)
        mistral_orig = judge_response(mistral_model, mistral_tokenizer, instruction, response)
        
        # Perturbations
        t = add_typos(response, seed=RANDOM_SEED)
        f = add_filler(response, seed=RANDOM_SEED)
        b = change_formatting(response)
        s = shuffle_sentences(response, seed=RANDOM_SEED)
        
        # Perturbed scores
        llama_typos = judge_response(llama_model, llama_tokenizer, instruction, t)
        llama_filler = judge_response(llama_model, llama_tokenizer, instruction, f)
        llama_bullets = judge_response(llama_model, llama_tokenizer, instruction, b)
        llama_shuffled = judge_response(llama_model, llama_tokenizer, instruction, s)
        
        mistral_typos = judge_response(mistral_model, mistral_tokenizer, instruction, t)
        mistral_filler = judge_response(mistral_model, mistral_tokenizer, instruction, f)
        mistral_bullets = judge_response(mistral_model, mistral_tokenizer, instruction, b)
        mistral_shuffled = judge_response(mistral_model, mistral_tokenizer, instruction, s)
        
        results.append({
            "id": i,
            "llama_orig": llama_orig,
            "llama_typos": llama_typos,
            "llama_filler": llama_filler,
            "llama_bullets": llama_bullets,
            "llama_shuffled": llama_shuffled,
            "mistral_orig": mistral_orig,
            "mistral_typos": mistral_typos,
            "mistral_filler": mistral_filler,
            "mistral_bullets": mistral_bullets,
            "mistral_shuffled": mistral_shuffled,
        })
        
        if (i + 1) % 10 == 0:
            print(f"Done {i+1}/{len(data)}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results/judge_consistency_100.csv", index=False)
    print("\\nResults saved to results/judge_consistency_100.csv")
    
    return df

# ============================================
# ANALYSIS
# ============================================

def compute_metrics(df):
    """Compute all consistency metrics."""
    
    def _metrics(orig, perturbed, perturbation_name, model_name):
        pairs = [(o, p) for o, p in zip(orig, perturbed) if pd.notna(o) and pd.notna(p)]
        orig_scores = [p[0] for p in pairs]
        pert_scores = [p[1] for p in pairs]
        
        exact_match = sum(1 for o, p in pairs if o == p) / len(pairs)
        mean_abs_diff = np.mean([abs(o - p) for o, p in pairs])
        spearman, _ = spearmanr(orig_scores, pert_scores)
        kendall, _ = kendalltau(orig_scores, pert_scores)
        flips = sum(1 for o, p in pairs if (o >= 4 and p <= 2) or (o <= 2 and p >= 4)) / len(pairs)
        
        # Statistical tests
        t_stat, t_pvalue = stats.ttest_rel(orig_scores, pert_scores)
        w_stat, w_pvalue = stats.wilcoxon(orig_scores, pert_scores)
        
        return {
            "model": model_name,
            "perturbation": perturbation_name,
            "exact_match": exact_match,
            "mean_abs_diff": mean_abs_diff,
            "spearman": spearman,
            "kendall": kendall,
            "category_flips": flips,
            "t_stat": t_stat,
            "t_pvalue": t_pvalue,
            "w_stat": w_stat,
            "w_pvalue": w_pvalue,
            "n": len(pairs)
        }
    
    metrics = []
    perturbations = ['typos', 'filler', 'bullets', 'shuffled']
    
    for pert in perturbations:
        metrics.append(_metrics(df[f"llama_orig"], df[f"llama_{pert}"], pert.capitalize(), "Llama 3.1"))
        metrics.append(_metrics(df[f"mistral_orig"], df[f"mistral_{pert}"], pert.capitalize(), "Mistral"))
    
    summary = pd.DataFrame(metrics)
    summary.to_csv("results/consistency_summary.csv", index=False)
    print("\\nSummary saved to results/consistency_summary.csv")
    
    return summary

if __name__ == "__main__":
    df = run_experiment()
    summary = compute_metrics(df)
    print("\\n=== SUMMARY ===")
    print(summary.to_string())
'''

with open('/content/llm-judge-robustness/src/experiment.py', 'w') as f:
    f.write(experiment_code)

print("experiment.py saved!")

viz_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("results/judge_consistency_100.csv")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================
# FIGURE 1: Exact Match Rate
# ============================================

perturbations = ['Typos', 'Filler', 'Bullets', 'Shuffled']
llama_matches = [
    (df['llama_orig'] == df['llama_typos']).mean(),
    (df['llama_orig'] == df['llama_filler']).mean(),
    (df['llama_orig'] == df['llama_bullets']).mean(),
    (df['llama_orig'] == df['llama_shuffled']).mean()
]
mistral_matches = [
    (df['mistral_orig'] == df['mistral_typos']).mean(),
    (df['mistral_orig'] == df['mistral_filler']).mean(),
    (df['mistral_orig'] == df['mistral_bullets']).mean(),
    (df['mistral_orig'] == df['mistral_shuffled']).mean()
]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(perturbations))
width = 0.35

bars1 = ax.bar(x - width/2, llama_matches, width, label='Llama 3.1 8B', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x + width/2, mistral_matches, width, label='Mistral 7B', color='#3498db', alpha=0.85)

ax.set_ylabel('Exact Match Rate', fontsize=12)
ax.set_xlabel('Perturbation Type', fontsize=12)
ax.set_title('Judge Consistency Under Output Perturbations', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(perturbations, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/figure1_exact_match.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 saved!")

# ============================================
# FIGURE 2 & 3: Heatmaps
# ============================================

for model, prefix in [('Llama 3.1', 'llama'), ('Mistral', 'mistral')]:
    for pert_name, pert_col in [('Typos', 'typos'), ('Shuffled', 'shuffled')]:
        fig, ax = plt.subplots(figsize=(8, 7))
        pairs = df[[f'{prefix}_orig', f'{prefix}_{pert_col}']].dropna()
        confusion = pd.crosstab(pairs[f'{prefix}_orig'], pairs[f'{prefix}_{pert_col}'], normalize='index')
        
        sns.heatmap(confusion, annot=True, fmt='.2%', cmap='Reds',
                   xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5],
                   cbar_kws={'label': 'Proportion'}, ax=ax)
        ax.set_xlabel(f'Perturbed Score ({pert_name})', fontsize=12)
        ax.set_ylabel('Original Score', fontsize=12)
        ax.set_title(f'{model} Score Changes Under {pert_name}', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'figures/figure_{prefix}_{pert_col}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

print("Heatmaps saved!")

# ============================================
# FIGURE 4: Category Flips
# ============================================

def category_flips(orig, perturbed):
    pairs = [(o, p) for o, p in zip(orig, perturbed) if pd.notna(o) and pd.notna(p)]
    flips = sum(1 for o, p in pairs if (o >= 4 and p <= 2) or (o <= 2 and p >= 4))
    return flips / len(pairs) if pairs else 0

llama_flips = [category_flips(df['llama_orig'], df[f'llama_{p}']) for p in ['typos', 'filler', 'bullets', 'shuffled']]
mistral_flips = [category_flips(df['mistral_orig'], df[f'mistral_{p}']) for p in ['typos', 'filler', 'bullets', 'shuffled']]

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, llama_flips, width, label='Llama 3.1 8B', color='#c0392b', alpha=0.85)
bars2 = ax.bar(x + width/2, mistral_flips, width, label='Mistral 7B', color='#2980b9', alpha=0.85)

ax.set_ylabel('Category Flip Rate', fontsize=12)
ax.set_xlabel('Perturbation Type', fontsize=12)
ax.set_title('Category Flip Rates (Good ↔ Bad)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(perturbations, fontsize=11)
ax.legend(fontsize=11)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/figure4_category_flips.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved!")
print("All figures generated!")
'''

with open('/content/llm-judge-robustness/src/visualize.py', 'w') as f:
    f.write(viz_code)

print("visualize.py saved!")

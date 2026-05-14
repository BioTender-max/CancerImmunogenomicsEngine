import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Simulate 150 tumors ──────────────────────────────────────────────────────
N_TUMORS = 150
N_HLA_LOCI = 6
HLA_LOCI = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRB1', 'HLA-DQB1', 'HLA-DPB1']

# TMB (mutations/Mb) — log-normal
tmb = np.random.lognormal(3.5, 1.0, N_TUMORS)

# Neoantigen burden (correlated with TMB)
neoantigen_burden = (tmb * np.random.uniform(0.1, 0.5, N_TUMORS)).astype(int) + 1

# MHC-I binding affinity (IC50 in nM, NetMHCpan-style)
# Strong binders: IC50 < 50 nM; Weak: 50-500; Non-binders: >500
n_peptides_per_tumor = 50
all_ic50 = []
for i in range(N_TUMORS):
    n_strong = int(neoantigen_burden[i] * 0.15)
    n_weak = int(neoantigen_burden[i] * 0.25)
    n_non = n_peptides_per_tumor - n_strong - n_weak
    ic50_strong = np.random.lognormal(np.log(20), 0.5, max(n_strong, 1))
    ic50_weak = np.random.lognormal(np.log(150), 0.4, max(n_weak, 1))
    ic50_non = np.random.lognormal(np.log(2000), 0.6, max(n_non, 1))
    all_ic50.extend(ic50_strong.tolist() + ic50_weak.tolist() + ic50_non.tolist())

all_ic50 = np.array(all_ic50[:N_TUMORS * n_peptides_per_tumor])
ic50_matrix = all_ic50.reshape(N_TUMORS, n_peptides_per_tumor)
mean_ic50 = ic50_matrix.mean(axis=1)

# HLA typing (4-digit resolution)
# Common HLA alleles per locus
hla_alleles = {
    'HLA-A': ['A*01:01', 'A*02:01', 'A*03:01', 'A*11:01', 'A*24:02', 'A*26:01'],
    'HLA-B': ['B*07:02', 'B*08:01', 'B*15:01', 'B*35:01', 'B*40:01', 'B*44:02'],
    'HLA-C': ['C*03:04', 'C*04:01', 'C*05:01', 'C*06:02', 'C*07:01', 'C*07:02'],
    'HLA-DRB1': ['DRB1*01:01', 'DRB1*03:01', 'DRB1*04:01', 'DRB1*07:01', 'DRB1*11:01', 'DRB1*15:01'],
    'HLA-DQB1': ['DQB1*02:01', 'DQB1*03:01', 'DQB1*05:01', 'DQB1*06:02'],
    'HLA-DPB1': ['DPB1*01:01', 'DPB1*02:01', 'DPB1*04:01', 'DPB1*04:02'],
}
hla_typing = {}
for locus in HLA_LOCI:
    alleles = hla_alleles[locus]
    hla_typing[locus] = np.random.choice(alleles, N_TUMORS)

# HLA diversity (heterozygosity per tumor)
hla_diversity = np.zeros(N_TUMORS)
for i in range(N_TUMORS):
    # Count unique alleles across loci
    allele_set = set()
    for locus in HLA_LOCI:
        allele_set.add(hla_typing[locus][i])
    hla_diversity[i] = len(allele_set) / N_HLA_LOCI

# Immunoediting detection (neoantigen depletion score)
# High TMB tumors with low neoantigen burden → immunoediting
expected_neoantigens = tmb * 0.3
observed_neoantigens = neoantigen_burden.astype(float)
immunoediting_score = 1 - (observed_neoantigens / (expected_neoantigens + 1e-10))
immunoediting_score = np.clip(immunoediting_score, -1, 1)

# Immune checkpoint expression (PD-L1/CTLA4/TIM3)
pdl1_expr = np.random.lognormal(1.5, 1.2, N_TUMORS)
ctla4_expr = np.random.lognormal(1.0, 1.0, N_TUMORS)
tim3_expr = np.random.lognormal(0.8, 1.1, N_TUMORS)
# Inflamed tumors have higher checkpoint expression
inflamed_mask = (tmb > np.percentile(tmb, 60)) & (neoantigen_burden > np.percentile(neoantigen_burden, 60))
pdl1_expr[inflamed_mask] *= np.random.uniform(2, 5, inflamed_mask.sum())
ctla4_expr[inflamed_mask] *= np.random.uniform(1.5, 3, inflamed_mask.sum())
tim3_expr[inflamed_mask] *= np.random.uniform(1.5, 3, inflamed_mask.sum())

# Tumor immune phenotype classification
immune_score = (np.log1p(pdl1_expr) + np.log1p(ctla4_expr) + np.log1p(tim3_expr)) / 3
phenotype = np.where(immune_score > np.percentile(immune_score, 67), 'Inflamed',
            np.where(immune_score > np.percentile(immune_score, 33), 'Excluded', 'Desert'))

n_inflamed = (phenotype == 'Inflamed').sum()
n_excluded = (phenotype == 'Excluded').sum()
n_desert = (phenotype == 'Desert').sum()

# HLA loss of heterozygosity
hla_loh = np.random.binomial(1, 0.15, N_TUMORS).astype(bool)
# LOH more common in immunoedited tumors
high_edit = immunoediting_score > 0.3
hla_loh[high_edit] = np.random.binomial(1, 0.35, high_edit.sum()).astype(bool)

# Allele frequencies for HLA-A
hla_a_alleles = hla_alleles['HLA-A']
hla_a_counts = {a: (hla_typing['HLA-A'] == a).sum() for a in hla_a_alleles}

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Cancer Immunogenomics Analysis Dashboard', fontsize=18,
             color='white', fontweight='bold', y=0.98)

DARK = '#161b22'
TEXT = 'white'
ACCENT = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# 1. Neoantigen burden distribution
ax = axes[0, 0]
style_ax(ax, '1. Neoantigen Burden Distribution')
ax.hist(neoantigen_burden, bins=30, color=ACCENT, edgecolor='#0d1117', alpha=0.85)
ax.axvline(np.median(neoantigen_burden), color='yellow', lw=2, ls='--',
           label=f'Median: {np.median(neoantigen_burden):.0f}')
ax.set_xlabel('Neoantigen Burden (count)', color=TEXT, fontsize=9)
ax.set_ylabel('Number of Tumors', color=TEXT, fontsize=9)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 2. MHC binding affinity distribution
ax = axes[0, 1]
style_ax(ax, '2. MHC-I Binding Affinity (IC50)')
ic50_flat = ic50_matrix.flatten()
ic50_log = np.log10(ic50_flat + 1)
ax.hist(ic50_log, bins=50, color=ACCENT, edgecolor='#0d1117', alpha=0.85)
ax.axvline(np.log10(50), color=ACCENT3, lw=2, ls='--', label='Strong binder (<50 nM)')
ax.axvline(np.log10(500), color='yellow', lw=2, ls='--', label='Weak binder (<500 nM)')
ax.set_xlabel('log10(IC50 nM)', color=TEXT, fontsize=9)
ax.set_ylabel('Count', color=TEXT, fontsize=9)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)
n_strong = (ic50_flat < 50).sum()
n_weak = ((ic50_flat >= 50) & (ic50_flat < 500)).sum()
ax.text(0.97, 0.95, f'Strong: {n_strong/len(ic50_flat)*100:.1f}%\nWeak: {n_weak/len(ic50_flat)*100:.1f}%',
        transform=ax.transAxes, ha='right', va='top', color=TEXT, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

# 3. HLA diversity
ax = axes[0, 2]
style_ax(ax, '3. HLA Allele Diversity')
ax.hist(hla_diversity, bins=20, color=ACCENT3, edgecolor='#0d1117', alpha=0.85)
ax.set_xlabel('HLA Diversity Score', color=TEXT, fontsize=9)
ax.set_ylabel('Number of Tumors', color=TEXT, fontsize=9)
ax.axvline(hla_diversity.mean(), color='yellow', lw=2, ls='--',
           label=f'Mean: {hla_diversity.mean():.2f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)
# HLA-A allele frequency bar
ax2_inset = ax.inset_axes([0.55, 0.4, 0.42, 0.55])
ax2_inset.set_facecolor('#21262d')
allele_labels = [a.split('*')[1] for a in hla_a_alleles]
allele_freqs = [hla_a_counts[a] / N_TUMORS for a in hla_a_alleles]
ax2_inset.bar(range(len(allele_labels)), allele_freqs, color=ACCENT, alpha=0.8)
ax2_inset.set_xticks(range(len(allele_labels)))
ax2_inset.set_xticklabels(allele_labels, fontsize=5, rotation=45, color=TEXT)
ax2_inset.tick_params(colors=TEXT, labelsize=5)
ax2_inset.set_title('HLA-A Freq', color=TEXT, fontsize=6)

# 4. Immunoediting score
ax = axes[1, 0]
style_ax(ax, '4. Immunoediting Detection Score')
colors_edit = [ACCENT2 if s > 0.3 else ACCENT for s in immunoediting_score]
ax.scatter(tmb, immunoediting_score, c=colors_edit, alpha=0.6, s=20, edgecolors='none')
ax.axhline(0.3, color='yellow', lw=1.5, ls='--', label='Immunoediting threshold')
ax.set_xlabel('TMB (mut/Mb)', color=TEXT, fontsize=9)
ax.set_ylabel('Immunoediting Score', color=TEXT, fontsize=9)
n_edited = (immunoediting_score > 0.3).sum()
patches = [mpatches.Patch(color=ACCENT2, label=f'Immunoedited (n={n_edited})'),
           mpatches.Patch(color=ACCENT, label=f'Non-edited (n={N_TUMORS-n_edited})')]
ax.legend(handles=patches, fontsize=8, facecolor='#21262d', labelcolor=TEXT)

# 5. Checkpoint expression
ax = axes[1, 1]
style_ax(ax, '5. Immune Checkpoint Expression')
checkpoints = ['PD-L1', 'CTLA-4', 'TIM-3']
expr_data = [np.log1p(pdl1_expr), np.log1p(ctla4_expr), np.log1p(tim3_expr)]
bp = ax.boxplot(expr_data, patch_artist=True, medianprops=dict(color='white', lw=2))
colors_cp = [ACCENT2, ACCENT, ACCENT3]
for patch, color in zip(bp['boxes'], colors_cp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]:
        item.set_color('#8b949e')
ax.set_xticklabels(checkpoints, color=TEXT, fontsize=9)
ax.set_ylabel('log(Expression + 1)', color=TEXT, fontsize=9)

# 6. Immune phenotype classification
ax = axes[1, 2]
style_ax(ax, '6. Tumor Immune Phenotype Classification')
phenotype_counts = [n_inflamed, n_excluded, n_desert]
phenotype_labels = ['Inflamed', 'Excluded', 'Desert']
colors_pheno = [ACCENT3, 'yellow', ACCENT2]
wedges, texts, autotexts = ax.pie(phenotype_counts, labels=phenotype_labels,
                                   autopct='%1.1f%%', colors=colors_pheno,
                                   textprops={'color': TEXT, 'fontsize': 10},
                                   startangle=90)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(9)

# 7. Neoantigen-TMB correlation
ax = axes[2, 0]
style_ax(ax, '7. Neoantigen Burden vs TMB Correlation')
colors_pheno_scatter = {'Inflamed': ACCENT3, 'Excluded': 'yellow', 'Desert': ACCENT2}
for ph in ['Inflamed', 'Excluded', 'Desert']:
    mask = phenotype == ph
    ax.scatter(tmb[mask], neoantigen_burden[mask], c=colors_pheno_scatter[ph],
               alpha=0.6, s=20, label=ph, edgecolors='none')
# Correlation
r, p = stats.pearsonr(np.log1p(tmb), np.log1p(neoantigen_burden))
ax.set_xlabel('TMB (mut/Mb)', color=TEXT, fontsize=9)
ax.set_ylabel('Neoantigen Burden', color=TEXT, fontsize=9)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor=TEXT)
ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}', transform=ax.transAxes,
        va='top', color=TEXT, fontsize=9, bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

# 8. HLA loss of heterozygosity
ax = axes[2, 1]
style_ax(ax, '8. HLA Loss of Heterozygosity')
loh_by_phenotype = {}
for ph in ['Inflamed', 'Excluded', 'Desert']:
    mask = phenotype == ph
    loh_by_phenotype[ph] = hla_loh[mask].mean() * 100
bars = ax.bar(list(loh_by_phenotype.keys()), list(loh_by_phenotype.values()),
              color=[ACCENT3, 'yellow', ACCENT2], edgecolor='#0d1117', alpha=0.85)
for bar, val in zip(bars, loh_by_phenotype.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
ax.set_ylabel('HLA LOH Rate (%)', color=TEXT, fontsize=9)
ax.set_ylim(0, max(loh_by_phenotype.values()) * 1.3)
ax.text(0.5, 0.9, f'Overall LOH: {hla_loh.mean()*100:.1f}%', transform=ax.transAxes,
        ha='center', color='yellow', fontsize=9)

# 9. Summary
ax = axes[2, 2]
style_ax(ax, '9. Analysis Summary')
ax.axis('off')
summary_lines = [
    ('Tumors Analyzed', f'{N_TUMORS}'),
    ('HLA Loci Typed', f'{N_HLA_LOCI}'),
    ('Median Neoantigen Burden', f'{np.median(neoantigen_burden):.0f}'),
    ('Strong MHC Binders', f'{(ic50_flat < 50).mean()*100:.1f}%'),
    ('Immunoedited Tumors', f'{n_edited} ({n_edited/N_TUMORS*100:.1f}%)'),
    ('Inflamed Phenotype', f'{n_inflamed} ({n_inflamed/N_TUMORS*100:.1f}%)'),
    ('Excluded Phenotype', f'{n_excluded} ({n_excluded/N_TUMORS*100:.1f}%)'),
    ('Desert Phenotype', f'{n_desert} ({n_desert/N_TUMORS*100:.1f}%)'),
    ('HLA LOH Rate', f'{hla_loh.mean()*100:.1f}%'),
    ('Neo-TMB Correlation', f'r={r:.3f}'),
]
y_pos = 0.95
for label, value in summary_lines:
    ax.text(0.05, y_pos, label + ':', color='#8b949e', fontsize=9, transform=ax.transAxes)
    ax.text(0.65, y_pos, value, color=ACCENT3, fontsize=9, fontweight='bold', transform=ax.transAxes)
    y_pos -= 0.09

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/cancer_immunogenomics_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()

import shutil
try:
    shutil.copy(__file__, '/mnt/shared-workspace/shared/cancer_immunogenomics_engine.py')
except shutil.SameFileError:
    pass  # already in destination

print("=== CancerImmunogenomicsEngine Results ===")
print(f"Tumors: {N_TUMORS}, HLA loci: {N_HLA_LOCI}")
print(f"Median neoantigen burden: {np.median(neoantigen_burden):.0f}")
print(f"Strong MHC binders (<50nM): {(ic50_flat < 50).mean()*100:.1f}%")
print(f"Immunoedited tumors: {n_edited} ({n_edited/N_TUMORS*100:.1f}%)")
print(f"Immune phenotypes - Inflamed: {n_inflamed}, Excluded: {n_excluded}, Desert: {n_desert}")
print(f"HLA LOH rate: {hla_loh.mean()*100:.1f}%")
print(f"Neoantigen-TMB correlation: r={r:.3f}, p={p:.2e}")
print(f"Dashboard saved: /mnt/shared-workspace/shared/cancer_immunogenomics_engine_dashboard.png")

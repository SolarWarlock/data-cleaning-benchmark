"""
CLI entrypoint to generate all 4 publication-quality figures:
Figure 1: Experimental pipeline architecture (ETL / ELT branches & dual-test evaluation)
Figure 2: Dual-test evaluation of LightGBM on clean test D* vs dirty stream D_test^dirty (r >= 20%)
Figure 3: Heatmap of break-even index (BEI) across defect types and noise ratios
Figure 4: Pareto frontier in coordinates "CPU cleaning time - F1 gain (Delta F1)"
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "experiments_results.parquet"

NOISE_LABELS_RU = {
    'mcar': 'Пропуски MCAR',
    'mnar': 'Пропуски MNAR',
    'outliers': 'Выбросы',
    'label_symmetric': 'Симм. шум меток',
    'label_asymmetric': 'Асимм. шум меток',
}

STRATEGY_LABELS_RU = {
    'baseline': 'Baseline',
    'simple': 'Simple IQR',
    'isolation_forest': 'Isolation Forest',
    'cleanlab': 'Cleanlab',
}

MODEL_LABELS_RU = {
    'lightgbm': 'LightGBM',
    'mlp': 'MLP',
    'logistic_regression': 'LogReg',
}

def generate_fig1_pipeline(output_path: Path):
    fig, ax = plt.subplots(figsize=(14.6, 5.4), dpi=300)
    ax.axis('off')
    title_text = "Архитектура экспериментального конвейера исследования, сопоставления ETL / ELT и двухконтурной оценки моделей"
    ax.text(0.5, 0.97, title_text, ha='center', va='top', fontsize=11.5, fontweight='bold', color='#000000')

    PAD = 0.006
    def draw_card(x, y, w, h, header_text, body_lines, bgcolor='#ffffff', bordercolor='#000000',
                  header_color='#000000', rx=0.012, is_container=False, b_sz=6.5, h_sz=7.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size={rx}",
                             facecolor=bgcolor, edgecolor=bordercolor, lw=1.3, zorder=2)
        ax.add_patch(box)
        if not is_container:
            ax.text(x + w/2, y + h - 0.035, header_text, ha='center', va='center',
                    fontsize=h_sz, fontweight='bold', color=header_color, zorder=3)
            ax.plot([x + 0.008, x + w - 0.008], [y + h - 0.060, y + h - 0.060],
                    color='#9ca3af', lw=0.7, zorder=3)
            n_lines = len(body_lines)
            if n_lines > 0:
                line_spacing = (h - 0.085) / max(n_lines, 1)
                for idx, line in enumerate(body_lines):
                    y_pos = y + h - 0.088 - idx * line_spacing
                    ax.text(x + w/2, y_pos, line, ha='center', va='center',
                            fontsize=b_sz, color='#111827', zorder=3)
        return {'left': x - PAD, 'right': x + w + PAD, 'bottom': y - PAD, 'top': y + h + PAD, 'cx': x + w/2, 'cy': y + h/2}

    def draw_arrow(x1, y1, x2, y2, color='#000000', lw=1.5, ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=12.5),
                    zorder=4)

    c_white, c_gray, c_border, c_head = '#ffffff', '#f3f4f6', '#111827', '#000000'

    b1 = draw_card(0.012, 0.47, 0.130, 0.40, "Исходные бенчмарки",
                   ["Adult Income", "Credit Fraud", "HELOC", "Hospital Readmission", "Разнородные домены"],
                   bgcolor=c_white, bordercolor=c_border, header_color=c_head, b_sz=6.5, h_sz=7.2)
    b2 = draw_card(0.165, 0.47, 0.132, 0.40, "NoiseInjector",
                   ["Пропуски MCAR", "Пропуски MNAR", "Числовые выбросы", "Симм. шум меток", "Асимм. шум меток"],
                   bgcolor=c_gray, bordercolor=c_border, header_color=c_head, b_sz=6.5, h_sz=7.2)
    draw_arrow(b1['right'] + 0.001, 0.670, b2['left'] - 0.001, 0.670)

    b3 = draw_card(0.375, 0.65, 0.170, 0.22, "ETL: Однократная очистка",
                   ["Санация хранилища", "Кэширование данных",
                    r"$T_{\mathrm{total}} = T_{\mathrm{clean}} + K \cdot T_{\mathrm{train}}$",
                    r"$K^* = \lceil T_{\mathrm{clean}}(N_{\mathrm{all}}) / T_{\mathrm{clean}}(N_{\mathrm{train}}) \rceil$"],
                   bgcolor=c_white, bordercolor=c_border, header_color=c_head, b_sz=6.2, h_sz=7.2)
    b4 = draw_card(0.375, 0.25, 0.170, 0.22, "ELT: Очистка на лету",
                   ["Потоковая загрузка данных", "Очистка перед каждым пуском",
                    r"$T_{\mathrm{total}} = K \cdot T_{\mathrm{clean}} + K \cdot T_{\mathrm{train}}$"],
                   bgcolor=c_gray, bordercolor=c_border, header_color=c_head, b_sz=6.2, h_sz=7.2)

    draw_arrow(b2['right'] + 0.001, 0.745, b3['left'] - 0.001, 0.745, color='#000000', lw=1.6)
    draw_arrow(b2['right'] + 0.001, 0.500, b4['left'] - 0.001, 0.360, color='#000000', lw=1.6, ls='--')

    b5 = draw_card(0.575, 0.36, 0.146, 0.51, "Стратегии очистки",
                   [r"$C_0$: Baseline", r"$C_1$: Simple IQR", r"$C_2$: Isolation Forest", r"$C_3$: Cleanlab",
                    r"Обучение на $D_{\mathrm{train}}$", "Без удаления тестовых строк"],
                   bgcolor=c_white, bordercolor=c_border, header_color=c_head, b_sz=6.3, h_sz=7.2)
    draw_arrow(b3['right'] + 0.001, 0.745, b5['left'] - 0.001, 0.670, color='#000000', lw=1.4)
    draw_arrow(b4['right'] + 0.001, 0.360, b5['left'] - 0.001, 0.510, color='#000000', lw=1.4, ls='--')

    b6 = draw_card(0.745, 0.36, 0.118, 0.51, "Классификаторы",
                   ["LightGBM", "MLP", "LogReg", "960 прогонов", "Факторный план"],
                   bgcolor=c_gray, bordercolor=c_border, header_color=c_head, b_sz=6.5, h_sz=7.2)
    draw_arrow(b5['right'] + 0.001, 0.615, b6['left'] - 0.001, 0.615, color='#000000', lw=1.4)

    b7 = draw_card(0.885, 0.36, 0.098, 0.51, "Оценка",
                   ["Контур 1:", r"Чистый $D^*$", "(100% строк)", "Контур 2:", r"Грязный $D^{\mathrm{dirty}}$"],
                   bgcolor=c_white, bordercolor=c_border, header_color=c_head, b_sz=6.3, h_sz=7.2)
    draw_arrow(b6['right'] + 0.001, 0.615, b7['left'] - 0.001, 0.615, color='#000000', lw=1.4)

    b_bot = draw_card(0.012, 0.045, 0.971, 0.09, "", [], bgcolor=c_gray, bordercolor=c_border, is_container=True)
    ax.text(0.50, 0.090, "Многокритериальная оценка вычислительной окупаемости (BEI), граница Парето и решающие правила выбора",
            ha='center', va='center', fontsize=9.8, fontweight='bold', color='#000000', zorder=4)

    draw_arrow(b4['cx'], b4['bottom'] - 0.001, b4['cx'], b_bot['top'] + 0.001, color='#000000')
    draw_arrow(b7['cx'], b7['bottom'] - 0.001, b7['cx'], b_bot['top'] + 0.001, color='#000000')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Fig 1: {output_path}")

def generate_fig2_dual_test(df: pd.DataFrame, output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300, sharey=True)
    lgb = df[df['model'] == 'lightgbm'].copy()
    noise_types = ["label_asymmetric", "label_symmetric", "mcar", "mnar", "outliers"]
    noise_ru_ticks = [NOISE_LABELS_RU.get(nt, nt) for nt in noise_types]
    x = np.arange(len(noise_types))
    bar_width = 0.35

    g_clean = lgb.groupby(['strategy', 'noise_type'])['clean_f1'].mean().unstack(level=0)
    base_clean = g_clean['baseline'].reindex(noise_types).values
    cleanlab_clean = g_clean['cleanlab'].reindex(noise_types).values

    lgb_dirty = lgb[lgb['noise_ratio'] >= 0.20]
    g_dirty = lgb_dirty.groupby(['strategy', 'noise_type'])['dirty_f1'].mean().unstack(level=0)
    base_dirty = g_dirty['baseline'].reindex(noise_types).values
    cleanlab_dirty = g_dirty['cleanlab'].reindex(noise_types).values

    c_base, c_clean = "#64748b", "#2563eb"

    ax1.bar(x - bar_width/2, base_clean, bar_width, label="Baseline", facecolor=c_base, edgecolor="#000000", linewidth=0.9)
    ax1.bar(x + bar_width/2, cleanlab_clean, bar_width, label="Cleanlab", facecolor=c_clean, edgecolor="#000000", linewidth=0.9)
    ax1.set_title("а) Чистый эталонный тест $D^*$ (без дефектов)", fontsize=10.5, pad=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(noise_ru_ticks, rotation=18, ha='right', fontsize=8.5)
    ax1.set_ylabel("F1-мера (macro)", fontsize=9.5)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', framealpha=0.9)

    ax2.bar(x - bar_width/2, base_dirty, bar_width, label="Baseline", facecolor=c_base, edgecolor="#000000", linewidth=0.9)
    ax2.bar(x + bar_width/2, cleanlab_dirty, bar_width, label="Cleanlab", facecolor=c_clean, edgecolor="#000000", linewidth=0.9)
    ax2.set_title("б) Реальный грязный поток $D_{\mathrm{test}}^{\mathrm{dirty}}$ ($r \ge 20\%$)", fontsize=10.5, pad=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(noise_ru_ticks, rotation=18, ha='right', fontsize=8.5)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated Fig 2: {output_path}")

def generate_fig3_heatmap(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)
    lgb = df[df['model'] == 'lightgbm'].copy()
    g = lgb.groupby(['noise_type', 'noise_ratio'])['break_even_ratio'].mean().unstack(level=1)
    noise_order = ["outliers", "mcar", "mnar", "label_symmetric", "label_asymmetric"]
    g = g.reindex([n for n in noise_order if n in g.index])
    g.index = [NOISE_LABELS_RU.get(idx, idx) for idx in g.index]
    g.columns = [f"{int(c*100)}%" for c in g.columns]

    sns.heatmap(g, annot=True, fmt=".3f", cmap="Blues", cbar_kws={'label': 'BEI (ΔF1 / с)'},
                linewidths=1.0, linecolor='#cbd5e1', ax=ax, annot_kws={"size": 10.5, "weight": "bold"})
    ax.set_title("Усредненный показатель удельной вычислительной отдачи (BEI)
для классификатора LightGBM по механизмам дефектов",
                 fontsize=10.5, pad=12, fontweight='bold')
    ax.set_xlabel("Интенсивность дефектов ($r$)", fontsize=10, labelpad=8)
    ax.set_ylabel("Механизм дефектов данных", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated Fig 3: {output_path}")

def generate_fig4_pareto(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=300)
    palette = {
        'baseline': '#10b981',
        'simple': '#f59e0b',
        'isolation_forest': '#8b5cf6',
        'cleanlab': '#2563eb'
    }
    markers = {'lightgbm': 'o', 'mlp': 's', 'logistic_regression': '^'}

    g = df.groupby(['strategy', 'model'], as_index=False).agg({
        'cleaning_cpu_sec': 'mean',
        'delta_f1': 'mean'
    })

    for strat in ['baseline', 'simple', 'isolation_forest', 'cleanlab']:
        for mod in ['lightgbm', 'mlp', 'logistic_regression']:
            sub = g[(g['strategy'] == strat) & (g['model'] == mod)]
            if len(sub) > 0:
                ax.scatter(sub['cleaning_cpu_sec'], sub['delta_f1'],
                           color=palette[strat], marker=markers[mod], s=95,
                           edgecolor='#000000', linewidth=1.1, zorder=5,
                           label=f"{STRATEGY_LABELS_RU[strat]} ({MODEL_LABELS_RU[mod]})")

    # Pareto line (Baseline -> Cleanlab LightGBM)
    base_lgb = g[(g['strategy'] == 'baseline') & (g['model'] == 'lightgbm')].iloc[0]
    clean_lgb = g[(g['strategy'] == 'cleanlab') & (g['model'] == 'lightgbm')].iloc[0]
    pareto_x = [base_lgb['cleaning_cpu_sec'], base_lgb['cleaning_cpu_sec'], clean_lgb['cleaning_cpu_sec']]
    pareto_y = [base_lgb['delta_f1'], clean_lgb['delta_f1'], clean_lgb['delta_f1']]
    ax.step(pareto_x, pareto_y, where='post', color='#dc2626', linestyle='--', linewidth=2.0, zorder=4, label='Граница Парето')

    ax.set_xscale('log')
    ax.set_xlabel("Процессорное время очистки $T_{\mathrm{clean}}$ (с, логарифмическая шкала)", fontsize=10)
    ax.set_ylabel("Прирост F1-меры ($\Delta F_1$)", fontsize=10)
    ax.set_title("Граница Парето в координатах «Процессорное время очистки — Прирост F1»", fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#000000", fontsize=7.2, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated Fig 4: {output_path}")

def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        sys.exit(1)
    df = pd.read_parquet(DATA_PATH)
    generate_fig1_pipeline(FIGURES_DIR / "experimental_pipeline_architecture.png")
    generate_fig2_dual_test(df, FIGURES_DIR / "dual_test_paradox.png")
    generate_fig3_heatmap(df, FIGURES_DIR / "break_even_heatmap.png")
    generate_fig4_pareto(df, FIGURES_DIR / "pareto_frontier.png")
    print("All figures successfully created in:", FIGURES_DIR)

if __name__ == "__main__":
    main()

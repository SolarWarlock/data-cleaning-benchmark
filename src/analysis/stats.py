"""Statistical analysis and publication-quality visualization module with Russian localization."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configure matplotlib for academic publication quality (VAK / Russian technical standards)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Academic Russian translation dictionaries
NOISE_LABELS_RU: Dict[str, str] = {
    "mcar": "Пропуски MCAR",
    "mnar": "Пропуски MNAR",
    "outliers": "Выбросы",
    "label_symmetric": "Симметричный шум меток",
    "label_asymmetric": "Асимметричный шум меток",
}

STRATEGY_LABELS_RU: Dict[str, str] = {
    "baseline": "Базовый уровень",
    "simple": "Винзоризация (Simple IQR)",
    "isolation_forest": "Изолирующий лес (IsoForest)",
    "cleanlab": "Фильтр Cleanlab",
}

MODEL_LABELS_RU: Dict[str, str] = {
    "lightgbm": "LightGBM",
    "mlp": "MLP (нейросеть)",
    "logistic_regression": "Логистическая регрессия",
}


class BenchmarkAnalyzer:
    """Analyzes benchmark experiment outputs, executes statistical tests, and builds figures in Russian."""

    def __init__(self, data_path: Path, output_dir: Optional[Path] = None) -> None:
        self.data_path = Path(data_path)
        self.output_dir = output_dir or Path("paper/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load experiment data from parquet or csv."""
        if self.data_path.suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        return pd.read_csv(self.data_path)

    def run_friedman_test(
        self, metric: str = "delta_f1"
    ) -> Tuple[float, float, pd.DataFrame]:
        """Execute Friedman test across cleaning strategies.

        Returns
        -------
        statistic : float
            Friedman test statistic.
        p_value : float
            Associated p-value.
        rank_df : pd.DataFrame
            Average ranks for each strategy.
        """
        pivot = self.df.pivot_table(
            index=["dataset", "noise_type", "noise_ratio", "model", "seed"],
            columns="strategy",
            values=metric,
            aggfunc="mean",
        ).dropna()

        strat_cols = [c for c in pivot.columns if c != "baseline"]
        samples = [pivot[col].values for col in strat_cols]

        stat, p_val = stats.friedmanchisquare(*samples)

        ranks = pivot[strat_cols].rank(axis=1, ascending=False).mean()
        rank_df = pd.DataFrame(
            {
                "Стратегия": [STRATEGY_LABELS_RU.get(s, s) for s in ranks.index],
                "Средний_ранг": ranks.values,
            }
        )

        return float(stat), float(p_val), rank_df

    def run_wilcoxon_tests(
        self, metric: str = "delta_f1"
    ) -> pd.DataFrame:
        """Run pairwise Wilcoxon signed-rank tests against baseline with Holm-Bonferroni correction."""
        pivot = self.df.pivot_table(
            index=["dataset", "noise_type", "noise_ratio", "model", "seed"],
            columns="strategy",
            values=metric,
            aggfunc="mean",
        ).dropna()

        results = []
        strat_cols = [c for c in pivot.columns if c != "baseline"]

        p_vals = []
        stats_list = []
        diff_means = []

        for strat in strat_cols:
            diffs = pivot[strat].values
            try:
                stat, p = stats.wilcoxon(diffs, zero_method="wilcox")
            except Exception:
                stat, p = 0.0, 1.0
            p_vals.append(p)
            stats_list.append(stat)
            diff_means.append(np.mean(diffs))

        # Holm-Bonferroni correction
        order = np.argsort(p_vals)
        corrected_p = np.empty(len(p_vals))
        m = len(p_vals)
        for i, idx in enumerate(order):
            corrected_p[idx] = min(p_vals[idx] * (m - i), 1.0)

        for i, strat in enumerate(strat_cols):
            results.append(
                {
                    "Стратегия": STRATEGY_LABELS_RU.get(strat, strat),
                    "Средняя_дельта": diff_means[i],
                    "W_статистика": stats_list[i],
                    "p_значение": p_vals[i],
                    "p_скорректированное": corrected_p[i],
                    "Значимо (alpha=0.05)": corrected_p[i] < 0.05,
                }
            )

        return pd.DataFrame(results)

    def generate_heatmap_break_even(
        self, model: str = "lightgbm", filename: str = "break_even_heatmap.pdf"
    ) -> Path:
        """Generate publication-ready heatmap of Break-Even Ratio with Russian text."""
        sub = self.df[self.df["model"] == model].copy()

        # Map noise types to Russian labels
        sub["noise_type_ru"] = sub["noise_type"].map(
            lambda x: NOISE_LABELS_RU.get(x, x)
        )

        pivot = sub.pivot_table(
            index="noise_type_ru",
            columns="noise_ratio",
            values="break_even_ratio",
            aggfunc="mean",
        )

        # Format column headers as percentage
        pivot.columns = [f"{int(c * 100)}%" if isinstance(c, (float, int)) else str(c) for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        cmap = sns.diverging_palette(10, 130, as_cmap=True)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            center=0.0,
            cbar_kws={"label": "Индекс удельной окупаемости (ΔF1 / CPU-сек)"},
            ax=ax,
            linewidths=0.5,
        )

        model_name_ru = MODEL_LABELS_RU.get(model, model)
        ax.set_title(f"Индекс окупаемости BEI для классификатора {model_name_ru}")
        ax.set_xlabel("Доля зашумления выборки")
        ax.set_ylabel("Тип внесенного искажения")

        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path

    def generate_pareto_plot(
        self, filename: str = "pareto_frontier.pdf"
    ) -> Path:
        """Plot cost (CPU seconds) versus quality gain (delta F1) with Russian labels."""
        fig, ax = plt.subplots(figsize=(7.0, 4.4))

        grouped = (
            self.df[self.df["strategy"] != "baseline"]
            .groupby(["strategy", "model"])[["cleaning_cpu_sec", "delta_f1"]]
            .mean()
            .reset_index()
        )

        palette = {"lightgbm": "#1f77b4", "mlp": "#ff7f0e", "logistic_regression": "#2ca02c"}
        markers = {"simple": "o", "isolation_forest": "s", "cleanlab": "^"}

        for _, row in grouped.iterrows():
            m = row["model"]
            strat = row["strategy"]
            m_ru = MODEL_LABELS_RU.get(m, m)
            s_ru = STRATEGY_LABELS_RU.get(strat, strat)

            ax.scatter(
                row["cleaning_cpu_sec"],
                row["delta_f1"],
                color=palette.get(m, "gray"),
                marker=markers.get(strat, "o"),
                s=100,
                label=f"{s_ru} [{m_ru}]",
                edgecolors="black",
                linewidth=0.5,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Среднее процессорное время очистки (с, логарифм. шкала)")
        ax.set_ylabel("Прирост качества классификации (ΔF1)")
        ax.set_title("Граница Парето: вычислительные затраты и прирост качества")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Build clean deduplicated legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="best",
            frameon=True,
            fontsize=8,
        )

        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path

    def generate_dual_test_paradox_plot(
        self, filename: str = "dual_test_paradox.pdf"
    ) -> Path:
        """Plot comparing Clean Test F1 versus Dirty Test F1 with Russian annotations."""
        fig, ax = plt.subplots(figsize=(7.2, 4.4))

        sub = self.df.dropna(subset=["clean_f1", "dirty_f1"]).copy()
        grouped = sub.groupby(["strategy", "noise_type"])[["clean_f1", "dirty_f1"]].mean().reset_index()

        bar_width = 0.35
        noise_types = grouped["noise_type"].unique()
        x = np.arange(len(noise_types))

        base_sub = grouped[grouped["strategy"] == "baseline"].set_index("noise_type").reindex(noise_types)
        clean_sub = grouped[grouped["strategy"] == "cleanlab"].set_index("noise_type").reindex(noise_types)

        if not base_sub.empty and not clean_sub.empty:
            ax.bar(
                x - bar_width / 2,
                base_sub["dirty_f1"],
                bar_width,
                label="Baseline (оценка на грязном тесте)",
                color="#7f7f7f",
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                x + bar_width / 2,
                clean_sub["dirty_f1"],
                bar_width,
                label="Cleanlab (оценка на грязном тесте)",
                color="#1f77b4",
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.set_xticks(x)
            noise_ru_ticks = [NOISE_LABELS_RU.get(nt, nt) for nt in noise_types]
            ax.set_xticklabels(noise_ru_ticks, rotation=12, ha="right")
            ax.set_ylabel("F1-мера на загрязненной тестовой выборке")
            ax.set_title("Сравнение устойчивости моделей при тестировании в загрязненной среде")
            ax.legend(loc="lower left", frameon=True)
            ax.grid(True, linestyle=":", alpha=0.5, axis="y")

        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path

    def generate_all_artifacts(self) -> Dict[str, Any]:
        """Compute all statistics and generate figures in Russian."""
        friedman_stat, friedman_p, ranks = self.run_friedman_test()
        wilcoxon_df = self.run_wilcoxon_tests()

        hm_path = self.generate_heatmap_break_even(model="lightgbm")
        hm_png = self.generate_heatmap_break_even(model="lightgbm", filename="break_even_heatmap.png")
        pareto_path = self.generate_pareto_plot()
        pareto_png = self.generate_pareto_plot(filename="pareto_frontier.png")
        paradox_path = self.generate_dual_test_paradox_plot()
        paradox_png = self.generate_dual_test_paradox_plot(filename="dual_test_paradox.png")

        return {
            "friedman_stat": friedman_stat,
            "friedman_p": friedman_p,
            "ranks": ranks,
            "wilcoxon": wilcoxon_df,
            "figures": {
                "heatmap_pdf": hm_path,
                "heatmap_png": hm_png,
                "pareto_pdf": pareto_path,
                "pareto_png": pareto_png,
                "paradox_pdf": paradox_path,
                "paradox_png": paradox_png,
            },
        }


if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer(
        data_path=Path("data/processed/experiments_results.parquet"),
        output_dir=Path("paper/figures"),
    )
    res = analyzer.generate_all_artifacts()
    print("=== АНАЛИЗ ЗАВЕРШЕН ДЛЯ ВСЕХ ЭКСПЕРИМЕНТОВ ===")
    print(f"Критерий Фридмана: chi2 = {res['friedman_stat']:.4f}, p = {res['friedman_p']:.4e}")
    print("\nПопарный критерий Вилкоксона (с поправкой Холма-Бонферрони):")
    print(res["wilcoxon"].to_string(index=False))
    print("\nСгенерированные иллюстрации (300 DPI, русский язык):")
    for k, v in res["figures"].items():
        print(f"  {k}: {v}")

"""
Gráfico da estrutura do WGT (peso amostral) -- evidência de que o peso identifica a classe de idade.

Uso (a partir da pasta com datasets/):
    python grafico_wgt.py
    python grafico_wgt.py --dados datasets --saida baseline_out/wgt_estrutura.png

Painel esquerdo: quantas pessoas partilham o mesmo WGT (treino+teste). Cada barra conta valores de WGT.
Painel direito: nos grupos com >=2 pessoas no treino, quantas classes de idade diferentes há dentro do grupo
                (amplitude = classe máxima - classe mínima).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, AZUL_ESC, CINZA = "#4c78a8", "#1f3f66", "#9aa5b1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="datasets")
    ap.add_argument("--saida", default=os.path.join("baseline_out", "wgt_estrutura.png"))
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(args.dados, "dataset_train.csv"), dtype=str)
    te = pd.read_csv(os.path.join(args.dados, "dataset_test_X.csv"), dtype=str)
    tr["AGE_RANGE"] = tr["AGE_RANGE"].astype(int)

    # ---- painel esquerdo: tamanho dos grupos de WGT (treino+teste) ----
    tam = pd.concat([tr["WGT"], te["WGT"]]).value_counts()          # WGT -> nº de pessoas
    dist = tam.value_counts().sort_index()                          # nº de pessoas -> nº de valores de WGT
    cap = 10
    barras = dist[dist.index <= cap].copy()
    barras.loc[cap + 1] = dist[dist.index > cap].sum()              # "11+"
    etiquetas = [str(i) for i in barras.index[:-1]] + [f"{cap + 1}+"]

    # ---- painel direito: amplitude de classes por grupo (treino) ----
    g = tr.groupby("WGT")["AGE_RANGE"].agg(["size", "min", "max"])
    g = g[g["size"] > 1]
    amp = (g["max"] - g["min"]).clip(upper=3)
    pct = amp.value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
    n_amp = amp.value_counts().reindex([0, 1, 2, 3], fill_value=0)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axs[0]
    ax.bar(etiquetas, barras.values, color=AZUL)
    for i, v in enumerate(barras.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("nº de pessoas com o mesmo WGT")
    ax.set_ylabel("nº de valores de WGT")
    ax.set_title(f"Quantas pessoas partilham o mesmo WGT?\n"
                 f"({tam.size} valores únicos em {int(tam.sum())} pessoas, treino+teste)", fontsize=11)

    ax = axs[1]
    cores = [AZUL_ESC, AZUL, CINZA, CINZA]
    ax.bar(["0", "1", "2", "3 ou mais"], pct.values, color=cores)
    for i, (p, n) in enumerate(zip(pct.values, n_amp.values)):
        ax.text(i, p, f"{p:.1f}%\n({int(n)} grupos)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(pct.max() * 1.2, 10))
    ax.set_xlabel("classes de idade diferentes dentro do mesmo WGT (máx. - mín.)")
    ax.set_ylabel("% dos grupos")
    ax.set_title(f"Quem partilha o WGT tem quase sempre a mesma classe de idade\n"
                 f"({len(g)} grupos com ≥2 pessoas no treino)", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    fig.savefig(args.saida, dpi=150)
    print(f"Guardado em {args.saida}")
    print(f"Grupos com amplitude 0: {pct[0]:.1f}% | amplitude média: {(g['max'] - g['min']).mean():.3f}")


if __name__ == "__main__":
    main()
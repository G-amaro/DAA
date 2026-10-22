"""
Exploração inicial -- AGE_RANGE (EHIS Portugal 2019)

Uso (a partir da pasta do repositório, onde existe datasets/):
    python exploracao_inicial.py            # grava os gráficos em figuras/
    python exploracao_inicial.py --show     # grava e também abre as janelas

Requisitos: numpy, pandas, matplotlib, seaborn, scikit-learn.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

TRAIN_PATH = "datasets/dataset_train.csv"
TEST_PATH = "datasets/dataset_test_X.csv"
FIG_DIR = "figuras"

ORD_HS1 = {"Muito mau": 1, "Mau": 2, "Razoável": 3, "Bom": 4, "Muito bom": 5}
ORD_HS3 = {"Nada limitado": 0, "Limitado mas não severamente": 1, "Severamente limitado": 2}

# colunas cujo "Não aplicável" funciona como filtro do questionário
BLOCOS = {
    "Trabalha (JOBISCO_COD)": "JOBISCO_COD",
    "Cuidados pessoais (PC1A)": "PC1A",
    "Tarefas domésticas (HA1A)": "HA1A",
    "Ajuda tarefas dom. (HA2)": "HA2",
    "Tabaco - histórico (IN45)": "IN45",
    "Álcool (AL2)": "AL2",
}


# --------------------------------------------------------------------------- #
# auxiliares
# --------------------------------------------------------------------------- #
def aplicavel(s):
    """True se a pergunta foi aplicada à pessoa (ou seja, NÃO é 'Não aplicável')."""
    return ~s.str.contains("não aplic", case=False, na=False)


def curto(s):
    p = str(s).split(" ")[0].strip(",")
    return {"Tem": "Empregado"}.get(p, p)


def codificar(s):
    """Para a informação mútua: numéricas com muitos valores -> decis; resto -> códigos."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= 10 and s.nunique() > 20:
        return pd.qcut(num, 10, duplicates="drop", labels=False).fillna(-1).astype(int)
    return pd.Series(pd.factorize(s.fillna("__nan__"))[0], index=s.index)


def guardar(fig_name, show):
    os.makedirs(FIG_DIR, exist_ok=True)
    caminho = os.path.join(FIG_DIR, fig_name)
    plt.savefig(caminho, dpi=130, bbox_inches="tight")
    print(f"  -> {caminho}")
    if show:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------- #
# gráficos
# --------------------------------------------------------------------------- #
def g1_alvo(y, show):
    """Como se distribui o alvo? (poucas pessoas nas classes extremas => Macro F1 sensível)"""
    cont = y.value_counts().sort_index()
    ax = cont.plot.bar(figsize=(8, 3.5), color="#4c78a8")
    ax.set_xlabel("AGE_RANGE"); ax.set_ylabel("nº de pessoas")
    ax.set_title("Distribuição do alvo (treino)")
    guardar("1_alvo.png", show)
    print("  proporções:", (cont / len(y)).round(3).to_dict())
    return cont


def g2_blocos(d, y, show):
    """Os blocos de 'Não aplicável' funcionam como filtro de idade?"""
    plt.figure(figsize=(9, 4.5))
    for nome, col in BLOCOS.items():
        r = aplicavel(d[col]).groupby(y).mean().sort_index()
        plt.plot(r.index, r.values, marker="o", label=nome)
    plt.xlabel("AGE_RANGE"); plt.ylabel("% a quem a pergunta se aplica")
    plt.title("Aplicabilidade dos blocos por classe"); plt.legend(fontsize=8)
    guardar("2_blocos_aplicabilidade.png", show)


def g3_mainstat(d, y, show):
    """Situação laboral por classe (onde saem os estudantes, onde dominam os reformados)."""
    tab = pd.crosstab(y, d["MAINSTAT"].map(curto), normalize="index")
    ax = tab.plot.bar(stacked=True, figsize=(10, 4.5), colormap="tab20", width=0.85)
    ax.set_ylabel("proporção"); ax.set_title("MAINSTAT por AGE_RANGE")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    guardar("3_mainstat.png", show)


def g4_saude(d, y, cd_cols, show):
    """Saúde percebida, limitações, IMC e nº de doenças por classe."""
    h = pd.to_numeric(d["BM1"], errors="coerce")
    w = pd.to_numeric(d["BM2"], errors="coerce")
    resumo = pd.DataFrame({
        "Saúde percebida (HS1, 1-5)": d["HS1"].map(ORD_HS1),
        "Limitação (HS3, 0-2)": d["HS3"].map(ORD_HS3),
        "% problema de longa duração (HS2)": (d["HS2"] == "Sim").astype(float),
        "IMC": w / (h / 100) ** 2,
        "Nº doenças crónicas (CD1A-P)": (d[cd_cols] == "Sim").sum(axis=1),
    }).groupby(y).mean().sort_index()

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes = axes.ravel()
    for ax, col in zip(axes, resumo.columns):
        ax.plot(resumo.index, resumo[col], marker="o")
        ax.set_title(col, fontsize=10); ax.set_xlabel("AGE_RANGE")
    axes[-1].axis("off")
    plt.tight_layout()
    guardar("4_saude_por_classe.png", show)


def g5_doencas(d, y, cd_cols, cont, show):
    """Que doenças crónicas aparecem com a idade?"""
    cd = (d[cd_cols] == "Sim").groupby(y).mean().sort_index()
    plt.figure(figsize=(11, 5))
    sns.heatmap(cd.T, cmap="viridis", cbar_kws={"label": "% com a doença"})
    plt.xlabel("AGE_RANGE"); plt.title("Doenças crónicas por classe")
    guardar("5_doencas_cronicas.png", show)
    prev = (cd.mul(cont.sort_index(), axis=0).sum() / len(y) * 100).round(1).sort_values()
    print("  prevalência global (%):", prev.to_dict())


def g6_wgt(d, y, show):
    """O peso amostral WGT esconde informação sobre a idade?"""
    if "WGT" not in d.columns:
        print("  WGT não existe neste ficheiro -- a saltar.")
        return
    wgt = pd.to_numeric(d["WGT"], errors="coerce")
    plt.figure(figsize=(9, 4))
    sns.boxplot(x=y, y=np.log10(wgt), color="#9ecae1")
    plt.xlabel("AGE_RANGE"); plt.ylabel("log10(WGT)"); plt.title("Peso amostral por classe")
    guardar("6_wgt.png", show)
    print("  Spearman(WGT, AGE_RANGE) =", round(wgt.corr(y, method="spearman"), 3))


def g7_informacao_mutua(d, y, show):
    """Que colunas informam realmente o alvo? (informação mútua; guia, não substitui um modelo)"""
    excluir = ["PID", "HHID", "AGE_RANGE", "WGT"]
    X = d.drop(columns=[c for c in excluir if c in d.columns])
    Xc = X.apply(codificar)
    mi = pd.Series(
        mutual_info_classif(Xc, y, discrete_features=True, random_state=0),
        index=Xc.columns,
    ).sort_values(ascending=False)

    mi.head(30)[::-1].plot.barh(figsize=(7, 8), color="#4c78a8")
    plt.xlabel("informação mútua com AGE_RANGE"); plt.title("Top 30 colunas")
    guardar("7_informacao_mutua.png", show)

    os.makedirs(FIG_DIR, exist_ok=True)
    mi.to_csv(os.path.join(FIG_DIR, "informacao_mutua.csv"), header=["MI"])
    fracas = mi[mi < 0.005]
    print(f"  {len(fracas)} colunas com MI < 0.005 (candidatas a descartar/agrupar):")
    print("  ", list(fracas.index))


def g8_pesos_por_classe(d, y, show):
    """Quota de cada classe na AMOSTRA vs ponderada por WGT.
    Os pesos são calibrados para a população, por isso a coluna 'ponderado' deve
    parecer-se com a pirâmide etária real -> serve para estimar as idades de cada classe."""
    if "WGT" not in d.columns:
        print("  WGT não existe neste ficheiro -- a saltar.")
        return
    w = pd.to_numeric(d["WGT"], errors="coerce")
    tab = pd.DataFrame({
        "amostra": y.value_counts(normalize=True).sort_index(),
        "ponderado": (w.groupby(y).sum() / w.sum()).sort_index(),
    })
    tab["acum_ponderado"] = tab["ponderado"].cumsum()
    print((tab * 100).round(1).to_string())
    os.makedirs(FIG_DIR, exist_ok=True)
    tab.to_csv(os.path.join(FIG_DIR, "quotas_por_classe.csv"))
    ax = tab[["amostra", "ponderado"]].plot.bar(figsize=(9, 4), color=["#9ecae1", "#3182bd"])
    ax.set_xlabel("AGE_RANGE"); ax.set_ylabel("proporção"); ax.set_title("Quota por classe: amostra vs ponderada (WGT)")
    guardar("8_quotas_ponderadas.png", show)


def g9_mi_condicional(d, y, show):
    """Informação mútua separada nas duas metades (classes 0-8 e 9-15):
    que colunas distinguem classes DENTRO de cada metade, depois de o corte do questionário ser conhecido."""
    excluir = ["PID", "HHID", "AGE_RANGE", "WGT"]
    X = d.drop(columns=[c for c in excluir if c in d.columns])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    saidas = {}
    for ax, (nome, mascara) in zip(axes, {"classes 0-8": y <= 8, "classes 9-15": y >= 9}.items()):
        Xc = X[mascara].apply(codificar)
        mi = pd.Series(
            mutual_info_classif(Xc, y[mascara], discrete_features=True, random_state=0),
            index=Xc.columns,
        ).sort_values(ascending=False)
        mi.head(25)[::-1].plot.barh(ax=ax, color="#4c78a8")
        ax.set_title(f"Top 25 dentro de {nome} (n={int(mascara.sum())})")
        ax.set_xlabel("informação mútua")
        saidas[nome] = mi
        print(f"  {nome}: top 12 ->", {k: round(v, 3) for k, v in mi.head(12).items()})
    plt.tight_layout()
    guardar("9_mi_condicional.png", show)
    os.makedirs(FIG_DIR, exist_ok=True)
    pd.DataFrame(saidas).to_csv(os.path.join(FIG_DIR, "informacao_mutua_condicional.csv"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="abrir também as janelas dos gráficos")
    args = ap.parse_args()

    sns.set_theme(style="whitegrid")
    # tudo como texto: as categorias vêm por extenso ("Razoável", "Sem resposta", ...)
    d_train = pd.read_csv(TRAIN_PATH, dtype=str)
    d_test = pd.read_csv(TEST_PATH, dtype=str)
    d_train["AGE_RANGE"] = d_train["AGE_RANGE"].astype(int)
    y = d_train["AGE_RANGE"]
    cd_cols = [c for c in d_train.columns if c.startswith("CD1")]

    print(d_train.shape, d_test.shape)
    print("colunas só no treino:", set(d_train.columns) - set(d_test.columns))
    print("colunas só no teste :", set(d_test.columns) - set(d_train.columns))

    print("\n[1] distribuição do alvo");              cont = g1_alvo(y, args.show)
    print("[2] blocos de 'Não aplicável'");           g2_blocos(d_train, y, args.show)
    print("[3] situação laboral");                    g3_mainstat(d_train, y, args.show)
    print("[4] saúde, IMC, nº de doenças");           g4_saude(d_train, y, cd_cols, args.show)
    print("[5] doenças crónicas");                    g5_doencas(d_train, y, cd_cols, cont, args.show)
    print("[6] peso amostral WGT");                   g6_wgt(d_train, y, args.show)
    print("[7] informação mútua");                    g7_informacao_mutua(d_train, y, args.show)
    print("[8] quotas por classe (amostra vs WGT)");  g8_pesos_por_classe(d_train, y, args.show)
    print("[9] informação mútua dentro de cada metade"); g9_mi_condicional(d_train, y, args.show)
    print(f"\nFeito. Gráficos em ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
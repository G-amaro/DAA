import argparse
import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedGroupKFold

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = "baseline_out"
SUB_DIR = "submissoes"
VIZ_COLS = ["viz_n", "viz_media", "viz_min", "viz_max", "viz_std"]
N_CLASSES = 16
CLASSES = np.arange(N_CLASSES)

IDENT = ["PID", "HHID", "AGE_RANGE"]
ADMIN = ["COUNTRY", "REFYEAR", "REFMONTH"]

ESCALAS = [
    {"Nenhuma dificuldade": 0, "Alguma dificuldade": 1, "Muita dificuldade": 2, "Não consegue de todo": 3},
    {"Nunca": 0, "Menos de uma vez por semana": 1, "1 a 3 vezes por semana": 2,
     "4 a 6 vezes por semana": 3, "Uma vez ou mais por dia": 4},
    {"Muito mau": 1, "Mau": 2, "Razoável": 3, "Bom": 4, "Muito bom": 5},
    {"Nada limitado": 0, "Limitado mas não severamente": 1, "Severamente limitado": 2},
    {"1º quintil": 1, "2º quintil": 2, "3º quintil": 3, "4º quintil": 4, "5º quintil": 5},
    {"Nenhuma": 0, "1 ou 2": 1, "3 a 5": 2, "6 ou mais": 3},
    {"Nenhum nível completado": 0, "Básico 1º e 2º ciclo": 1, "Básico 3º ciclo": 2, "Secundário": 3,
     "Pós secundário": 4, "C.T.Superior Profissional": 5, "Bacharelato ou licenciatura 1º ciclo": 6,
     "Licenciatura pré-bolonha ou mestrado 2º ciclo": 7, "Doutoramento": 8},
]


def _tem(s, texto):
    return s.str.contains(texto, case=False, na=False)


def construir_features(df, usar_wgt, wgt_extra):
    """df: treino+teste concatenados (tudo texto). Devolve X (numéricas + categóricas) e a flag 55+."""
    aplica55 = ~_tem(df["PC1A"], "não aplic")          # pergunta só feita a quem tem 55+ anos
    trabalha = ~_tem(df["JOBISCO_COD"], "não aplic")

    drop = IDENT + ADMIN + ([] if usar_wgt else ["WGT"])
    X = df.drop(columns=[c for c in drop if c in df.columns])

    cols = {}
    for c in X.columns:
        s = X[c]
        sr, na = _tem(s, "sem resposta"), _tem(s, "não aplic")
        real = s.mask(sr | na)                       # valores "a sério" (N/A e Sem resposta -> NaN)
        vals = set(real.dropna().unique())
        if c == "WGT":
            cols[c] = pd.to_numeric(s, errors="coerce")
        elif not vals:
            continue                                  # coluna sem informação
        elif vals <= {"Sim", "Não"}:
            cols[c] = real.map({"Sim": 1.0, "Não": 0.0})
        elif any(vals <= set(e) for e in ESCALAS):
            esc = next(e for e in ESCALAS if vals <= set(e))
            cols[c] = real.map(esc).astype(float)
        elif pd.to_numeric(real.dropna(), errors="coerce").notna().mean() > 0.9:
            cols[c] = pd.to_numeric(real, errors="coerce")
        else:                                         # categórica; "Não aplicável" fica como categoria
            cat = s.mask(sr)
            cols[c] = pd.Series(pd.Categorical(cat, categories=sorted(cat.dropna().unique())), index=s.index)
    X = pd.DataFrame(cols, index=df.index)

    def gnum(c):
        return X[c] if c in X.columns and is_numeric_dtype(X[c]) else pd.Series(np.nan, index=X.index)

    def soma(cs):
        return pd.concat([gnum(c) for c in cs], axis=1).sum(axis=1, min_count=1)

    cd_cols = [c for c in X.columns if c.startswith("CD1")]
    adl = soma(["PC1A", "PC1B", "PC1C", "PC1D", "PC1E"])
    iadl = soma([f"HA1{x}" for x in "ABCDEFG"])
    n_doencas = soma(cd_cols)
    h, w = gnum("BM1"), gnum("BM2")

    X["aplica_55mais"] = aplica55.astype(int)
    X["trabalha"] = trabalha.astype(int)
    X["imc"] = w / (h / 100) ** 2
    X["n_doencas"] = n_doencas
    X["adl_soma"] = adl
    X["iadl_soma"] = iadl
    deficits = pd.concat([
        (5 - gnum("HS1")) / 4,
        gnum("HS2"),
        gnum("HS3") / 2,
        (n_doencas / 8).clip(upper=1),
        adl / 15,
        iadl / 21,
    ], axis=1)
    X["fragilidade"] = deficits.mean(axis=1)          # média dos défices disponíveis (ignora NaN)

    # ---- features de desenho amostral (só com WGT e demografia, sem alvo) ----
    if usar_wgt and wgt_extra and "WGT" in df.columns:
        wgt = pd.to_numeric(df["WGT"], errors="coerce")
        lw = np.log(wgt.where(wgt > 0))
        base = [c for c in ["NUTS2_AG", "SEX", "DEG_URB"] if c in df.columns]
        if base:
            celula = df[base].astype(str).agg("|".join, axis=1)
        else:
            celula = pd.Series("x", index=df.index)
        X["logwgt"] = lw
        X["wgt_rel"] = lw - lw.groupby(celula).transform("mean")
        X["wgt_rank_cel"] = wgt.groupby(celula).rank(pct=True)
        X["wgt_dup"] = wgt.map(wgt.value_counts())
    return X, aplica55.values


# --------------------------------------------------------------------------- #
# modelos
# --------------------------------------------------------------------------- #
def parametros(rapido, base):
    p = dict(
        n_estimators=60 if rapido else 300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        random_state=0, n_jobs=-1, verbose=-1,
    )
    if not base:
        p.update(
            n_estimators=60 if rapido else 400, learning_rate=0.04, num_leaves=15, min_child_samples=40,
            colsample_bytree=0.4, reg_lambda=5.0, cat_smooth=50, min_data_per_group=100, cat_l2=10,
        )
    return p


def proba16(modelo, X):
    P = modelo.predict_proba(X)
    full = np.zeros((len(X), N_CLASSES))
    full[:, modelo.classes_.astype(int)] = P
    return full


# --------------------------------------------------------------------------- #
# codificação por WGT: classes das outras pessoas do treino com o mesmo WGT
# --------------------------------------------------------------------------- #
def enc_vizinhos(k_ref, y_ref, k_alvo):
    """Estatísticas da classe das linhas de referência com a mesma chave (WGT) que cada linha alvo."""
    g = pd.DataFrame({"k": np.asarray(k_ref), "y": np.asarray(y_ref, dtype=float)}).groupby("k")["y"].agg(
        viz_n="size", viz_media="mean", viz_min="min", viz_max="max", viz_std="std")
    out = g.reindex(pd.Index(np.asarray(k_alvo)))
    out["viz_n"] = out["viz_n"].fillna(0)
    return out[VIZ_COLS].reset_index(drop=True)


def enc_oof(k, y, n_splits=5, seed=0):
    """Igual a enc_vizinhos, mas para as próprias linhas de treino via folds internos (sem usar o próprio alvo)."""
    k, y = np.asarray(k), np.asarray(y)
    res = pd.DataFrame(np.nan, index=range(len(k)), columns=VIZ_COLS)
    res["viz_n"] = 0.0
    for i_ref, i_alvo in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(k):
        res.iloc[i_alvo] = enc_vizinhos(k[i_ref], y[i_ref], k[i_alvo]).values
    return res


def com_viz(Xtr, ytr, ktr, Xte, kte):
    """Junta as features viz_* a Xtr (via folds internos) e a Xte (via todo o treino)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    etr = enc_oof(ktr, ytr)
    ete = enc_vizinhos(ktr, ytr, kte)
    for c in VIZ_COLS:
        Xtr[c] = etr[c].values
        Xte[c] = ete[c].values
    return Xtr, Xte


def rodar(Xtr, ytr, ap_tr, Xte, ap_te, params, seeds, ktr=None, kte=None):
    """Treina as variantes em (Xtr, ytr) e devolve as probabilidades (média de seeds) para Xte.
    Se ktr/kte (chaves WGT) forem dadas, acrescenta as features de vizinhos de peso."""
    if ktr is not None:
        Xtr, Xte = com_viz(Xtr, ytr, ktr, Xte, kte)
    pu = np.zeros((len(Xte), N_CLASSES))
    pm = np.zeros((len(Xte), N_CLASSES))
    for s in range(seeds):
        p = dict(params, random_state=params["random_state"] + s)
        pu += proba16(lgb.LGBMClassifier(class_weight="balanced", **p).fit(Xtr, ytr), Xte)
        for h in (False, True):
            mt, me = ap_tr == h, ap_te == h
            if mt.sum() == 0 or me.sum() == 0:
                continue
            ytr_h = ytr[mt]
            if ytr_h.nunique() > 1:
                pm[me] += proba16(
                    lgb.LGBMClassifier(class_weight="balanced", **p).fit(Xtr[mt], ytr_h), Xte[me])
            else:
                pm[me, int(ytr_h.iloc[0])] += 1.0
    return {"proba_unico": pu / seeds, "proba_metades": pm / seeds}


# --------------------------------------------------------------------------- #
# métricas e multiplicadores por classe (Macro F1)
# --------------------------------------------------------------------------- #
def f1_rapido(y, p):
    tp = np.bincount(y[y == p], minlength=N_CLASSES)
    pc = np.bincount(p, minlength=N_CLASSES)
    yc = np.bincount(y, minlength=N_CLASSES)
    den = pc + yc
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def metricas(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "macro_f1": f1_rapido(y, pred),
        "exatidao": float((y == pred).mean()),
        "exatidao_+-1_classe": float((np.abs(y - pred) <= 1).mean()),
    }


def otimizar_pesos(P, y, passes=3):
    """Multiplicadores por classe sobre as probabilidades que maximizam o Macro F1 (subida por coordenadas)."""
    grade = np.exp(np.linspace(-1.2, 1.2, 25))
    w = np.ones(N_CLASSES)
    melhor = f1_rapido(y, (P * w).argmax(1))
    for _ in range(passes):
        for c in range(N_CLASSES):
            for g in grade:
                t = w.copy()
                t[c] = g
                v = f1_rapido(y, (P * t).argmax(1))
                if v > melhor + 1e-9:
                    melhor, w = v, t
    return w


def previsoes_honestas(P, y, seed=0):
    """Ajusta os multiplicadores numa metade das previsões OOF e prevê a outra (e vice-versa)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    a, b = perm[: len(y) // 2], perm[len(y) // 2:]
    pred = np.zeros(len(y), dtype=int)
    for treino, aval in ((a, b), (b, a)):
        w = otimizar_pesos(P[treino], y[treino])
        pred[aval] = (P[aval] * w).argmax(1)
    return pred


# --------------------------------------------------------------------------- #
# propagação dentro de grupos de WGT (linhas de avaliação que partilham o mesmo peso)
# --------------------------------------------------------------------------- #
MODOS = ("media", "soma")


def propagar(P, k, modo):
    """Combina as log-probabilidades das linhas que partilham o mesmo WGT (todas com a mesma classe, quase sempre).
    modo 'media': média das log-probabilidades do grupo (conservador, não conta duas vezes a mesma evidência).
    modo 'soma' : soma (produto de especialistas; assume independência entre as linhas do grupo).
    Linhas sozinhas no seu WGT ficam inalteradas. Só usa previsões, nunca labels."""
    k = np.asarray(k, dtype=object)
    k = np.where(pd.isna(k), np.array([f"__na{i}" for i in range(len(k))], dtype=object), k)
    g = pd.DataFrame(np.log(np.clip(P, 1e-6, 1.0))).groupby(k)
    S = (g.transform("mean") if modo == "media" else g.transform("sum")).values
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="poucas árvores (só para testar)")
    ap.add_argument("--sem-wgt", action="store_true", help="não usar WGT")
    ap.add_argument("--sem-wgt-extra", action="store_true", help="WGT bruto, sem features de desenho amostral")
    ap.add_argument("--params-base", action="store_true", help="hiperparâmetros do baseline original")
    ap.add_argument("--sem-viz", action="store_true", help="desliga a codificação por WGT (vizinhos de peso)")
    ap.add_argument("--sem-prop", action="store_true", help="desliga a propagação por grupos de WGT")
    ap.add_argument("--seeds", type=int, default=1, help="nº de seeds a promediar")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dados", default="datasets", help="pasta com dataset_train.csv e dataset_test_X.csv")
    args = ap.parse_args()
    params = parametros(args.rapido, args.params_base)

    tr = pd.read_csv(os.path.join(args.dados, "dataset_train.csv"), dtype=str)
    te = pd.read_csv(os.path.join(args.dados, "dataset_test_X.csv"), dtype=str)
    y = tr["AGE_RANGE"].astype(int).reset_index(drop=True)
    n_tr = len(tr)
    print(f"treino {tr.shape}, teste {te.shape}")

    usar_viz = (not args.sem_viz) and (not args.sem_wgt)
    usar_prop = (not args.sem_prop) and (not args.sem_wgt)
    k_tr, k_te = tr["WGT"].values, te["WGT"].values          # chave = valor exato do WGT (texto)
    if usar_viz:
        print(f"cobertura WGT: {np.isin(k_te, k_tr).mean():.1%} das linhas do teste têm o mesmo WGT "
              f"de alguma pessoa do treino; {pd.Series(k_tr).duplicated(keep=False).mean():.1%} das "
              f"linhas do treino partilham o WGT com outra linha do treino")

    todo = pd.concat([tr.drop(columns=["AGE_RANGE"]), te], ignore_index=True)
    X_all, ap_all = construir_features(todo, not args.sem_wgt, not args.sem_wgt_extra)
    X, Xte = X_all.iloc[:n_tr].reset_index(drop=True), X_all.iloc[n_tr:].reset_index(drop=True)
    ap_tr, ap_te = ap_all[:n_tr], ap_all[n_tr:]
    print(f"features: {X.shape[1]} (categóricas: {sum(str(t) == 'category' for t in X.dtypes)})")
    print(f"WGT: {'não' if args.sem_wgt else 'sim'} | features WGT extra: "
          f"{'não' if (args.sem_wgt or args.sem_wgt_extra) else 'sim'} | "
          f"params: {'base' if args.params_base else 'regularizados'} | seeds: {args.seeds} | "
          f"vizinhos WGT: {'sim' if usar_viz else 'não'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "categoricas.txt"), "w", encoding="utf-8") as f:
        for c in X.columns:
            if str(X[c].dtype) == "category":
                cats = list(X[c].cat.categories)
                f.write(f"{c} ({len(cats)}): {cats[:25]}{' ...' if len(cats) > 25 else ''}\n")

    # ---- validação cruzada (OOF) -------------------------------------------
    oof = {"proba_unico": np.zeros((n_tr, N_CLASSES)), "proba_metades": np.zeros((n_tr, N_CLASSES))}
    fold_id = np.zeros(n_tr, dtype=int)
    cov_oof = np.zeros(n_tr, dtype=bool)          # linha de validação com pelo menos um vizinho de WGT no treino do fold
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    for k, (i_tr, i_va) in enumerate(skf.split(X, y, groups=tr["HHID"].values), 1):
        r = rodar(X.iloc[i_tr], y.iloc[i_tr].reset_index(drop=True), ap_tr[i_tr],
                  X.iloc[i_va], ap_tr[i_va], params, args.seeds,
                  k_tr[i_tr] if usar_viz else None, k_tr[i_va] if usar_viz else None)
        for nome, v in r.items():
            oof[nome][i_va] = v
        fold_id[i_va] = k
        if not args.sem_wgt:
            cov_oof[i_va] = enc_vizinhos(k_tr[i_tr], y.values[i_tr], k_tr[i_va])["viz_n"].values > 0
        print(f"  fold {k}/{args.folds} feito")

    yv = y.values
    preds_oof = {
        "cls_unico": oof["proba_unico"].argmax(1),
        "cls_metades": oof["proba_metades"].argmax(1),
    }
    if usar_prop:                                  # a propagação faz-se dentro de cada fold, como no teste
        for nome, P in (("cls_unico", oof["proba_unico"]), ("cls_metades", oof["proba_metades"])):
            for modo in MODOS:
                Pp = np.zeros_like(P)
                for f in np.unique(fold_id):
                    m = fold_id == f
                    Pp[m] = propagar(P[m], k_tr[m], modo)
                preds_oof[f"{nome}_prop_{modo}"] = Pp.argmax(1)
    res = pd.DataFrame({n: metricas(yv, p) for n, p in preds_oof.items()}).T.sort_values("macro_f1", ascending=False)
    print("\nResultados (validação cruzada, previsões out-of-fold):")
    print(res.round(4).to_string())
    if usar_prop:
        print("(*_prop_*: previsões combinadas entre linhas de validação do mesmo fold que partilham o WGT)")

    melhor = res.index[0]
    f1c = f1_score(yv, preds_oof[melhor], labels=CLASSES, average=None, zero_division=0)
    print(f"\nF1 por classe da melhor variante ({melhor}):")
    print({int(c): round(float(v), 3) for c, v in zip(CLASSES, f1c)})

    if not args.sem_wgt:                           # diagnóstico: onde falha, por classe e por cobertura de vizinhos
        pm = preds_oof[melhor]
        print("\nDiagnóstico por cobertura de vizinhos de WGT (no CV cada fold vê 80% do treino; no teste vê 100%):")
        for nome, m in (("com vizinho", cov_oof), ("sem vizinho", ~cov_oof)):
            print(f"  {nome}: {m.mean():.1%} das linhas | macro F1 {f1_rapido(yv[m], pm[m]):.4f} "
                  f"| exatidão {(yv[m] == pm[m]).mean():.4f}")
        d = pd.DataFrame({"y": yv, "cov": cov_oof, "ok": yv == pm})
        diag = pd.DataFrame({
            "n": np.bincount(yv, minlength=N_CLASSES),
            "cobertura": d.groupby("y")["cov"].mean().reindex(CLASSES),
            "f1": f1c,
            "exat_com_viz": d[d["cov"]].groupby("y")["ok"].mean().reindex(CLASSES),
            "exat_sem_viz": d[~d["cov"]].groupby("y")["ok"].mean().reindex(CLASSES),
        }, index=CLASSES)
        diag.index.name = "classe"
        diag.to_csv(os.path.join(OUT_DIR, "diagnostico_classes.csv"))
        print(diag.round(3).T.to_string())

    # ---- modelo final em todo o treino -> teste ---------------------------------
    print("\nA treinar no treino completo e a prever o teste...")
    r = rodar(X, y, ap_tr, Xte, ap_te, params, args.seeds,
              k_tr if usar_viz else None, k_te if usar_viz else None)
    preds_te = {
        "cls_unico": r["proba_unico"].argmax(1),
        "cls_metades": r["proba_metades"].argmax(1),
    }
    if usar_prop:                                  # no teste, os grupos de WGT formam-se entre as linhas do teste
        for nome, P in (("cls_unico", r["proba_unico"]), ("cls_metades", r["proba_metades"])):
            for modo in MODOS:
                preds_te[f"{nome}_prop_{modo}"] = propagar(P, k_te, modo).argmax(1)
    for nome, p in preds_te.items():
        pd.DataFrame({"RowId": np.arange(len(te)), "AGE_RANGE": p}).to_csv(
            os.path.join(SUB_DIR, f"submission_{nome}.csv"), index=False)
    res.to_csv(os.path.join(OUT_DIR, "resultados_baseline.csv"))
    pd.DataFrame({
        "PID": tr["PID"], "AGE_RANGE": yv,
        "pred_" + melhor: preds_oof[melhor],
        "classe_esperada": oof["proba_metades"] @ CLASSES,       # "idade biológica" contínua (em classes)
        "desvio": oof["proba_metades"] @ CLASSES - yv,
    }).to_csv(os.path.join(OUT_DIR, "oof_predicoes.csv"), index=False)

    dist = pd.DataFrame({
        "treino": np.bincount(yv, minlength=N_CLASSES) / n_tr,
        f"previsto_teste_{melhor}": np.bincount(preds_te[melhor], minlength=N_CLASSES) / len(te),
    }).round(3)
    print("\nDistribuição por classe (treino real vs previsão no teste):")
    print(dist.T.to_string())

    # importância das features (modelo final, classificador único)
    X_imp = com_viz(X, y, k_tr, Xte, k_te)[0] if usar_viz else X
    imp = pd.Series(
        lgb.LGBMClassifier(class_weight="balanced", **params).fit(X_imp, y).booster_.feature_importance("gain"),
        index=X_imp.columns).sort_values(ascending=False)
    (imp / imp.sum()).to_csv(os.path.join(OUT_DIR, "importancia_features.csv"), header=["ganho_relativo"])
    print("\ntop 15 features (ganho):", {k: round(float(v), 3) for k, v in (imp / imp.sum()).head(15).items()})
    print(f"\nFeito. Submissões em ./{SUB_DIR}/ e resultados em ./{OUT_DIR}/")
    print(f"Sugestão para a próxima submissão no Kaggle: {SUB_DIR}/submission_{melhor}.csv")


if __name__ == "__main__":
    main()
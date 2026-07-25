"""
Identification pure du pattern de double rejet — AUCUNE simulation de trade.

Pattern :
    1. Un evenement above-ask (ou below-bid) se produit, culminant a B1.
    2. Il est rejete : le prix ne redepasse pas B1 pendant CONFIRM_SECS.
    3. Un second evenement, de meme sens, se produit, culminant a B2, avec
       B2 DEGRADE par rapport a B1 (plus bas pour un sweep acheteur, plus
       haut pour un sweep vendeur).
    4. Il est rejete lui aussi.
    5. L'ecart de prix entre B1 et B2 ne depasse pas GAP_MAX_POINTS points.
    6. Le delai entre la fin du premier et la fin du second ne depasse pas
       SEQ_MAX_SECS secondes.

Le volume n'intervient dans AUCUN critere de filtrage — il est seulement
affiche pour information. Ce script liste les occurrences, il ne mesure
aucune performance (pas d'entree, pas de stop, pas de MFE/MAE).

Usage :
    python identifier_rejet.py ticks_ES_full.csv
    python identifier_rejet.py ticks_ES_full.csv --sens sell --confirm 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NS = 1_000_000_000
TICK = 0.25

SESSION_START = "13:30"
SESSION_END   = "14:45"

GROUP_MS       = 50    # executions plus rapprochees que ca = un seul ordre
CONFIRM_SECS   = 3     # duree sans depassement pour valider un rejet
SEQ_MAX_SECS   = 120   # 2 minutes max entre la fin des deux evenements
GAP_MAX_POINTS = 3      # ecart max entre B1 et B2, en points d'indice


# ---------------------------------------------------------------------------
# Chargement + classification (identique a rejet_above.py)
# ---------------------------------------------------------------------------

def charger(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, names=["time", "price", "bid", "ask", "qty"])
    try:
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def classifier(df: pd.DataFrame) -> pd.DataFrame:
    spread_ok = df["ask"] > df["bid"]
    df["above"] = spread_ok & (df["price"] > df["ask"])
    df["below"] = spread_ok & (df["price"] < df["bid"])
    return df


# ---------------------------------------------------------------------------
# Regroupement en evenements (facon tape reconstruit)
# ---------------------------------------------------------------------------

def grouper(df: pd.DataFrame, masque, sens: str, group_ns: int) -> pd.DataFrame:
    sub = df[masque]
    if sub.empty:
        return pd.DataFrame()

    t = sub["time"].values.astype("datetime64[ns]").astype("int64")
    groupe = np.concatenate(([0], np.cumsum(np.diff(t) > group_ns)))

    g = sub.assign(_g=groupe).groupby("_g")
    ev = pd.DataFrame({
        "debut": g["time"].first(),
        "fin": g["time"].last(),
        "volume": g["qty"].sum(),
        "n_prints": g.size(),
        "prix_min": g["price"].min(),
        "prix_max": g["price"].max(),
    }).reset_index(drop=True)

    ev["B"] = ev["prix_max"] if sens == "buy" else ev["prix_min"]
    ev["niveaux"] = ((ev["prix_max"] - ev["prix_min"]) / TICK).round().astype(int) + 1
    return ev


# ---------------------------------------------------------------------------
# Rejet causal : le niveau tient-il pendant confirm_ns apres la fin ?
# ---------------------------------------------------------------------------

def confirmer(ev: pd.DataFrame, px: np.ndarray, tns: np.ndarray,
              sens: str, confirm_ns: int) -> list:
    rejets = []
    for r in ev.itertuples():
        fin_ns = pd.Timestamp(r.fin).value
        t_conf = fin_ns + confirm_ns
        lo = int(np.searchsorted(tns, fin_ns, side="right"))
        hi = int(np.searchsorted(tns, t_conf, side="right"))
        if hi <= lo:
            continue
        seg = px[lo:hi]
        depasse = (seg > r.B).any() if sens == "buy" else (seg < r.B).any()
        if not depasse:
            rejets.append(r)
    return rejets


# ---------------------------------------------------------------------------
# Appariement — SEUL critere de prix : ecart en points. Le volume n'entre
# dans rien ici, il est juste transporte pour affichage.
# ---------------------------------------------------------------------------

def identifier(rejets: list, sens: str, gap_max_points: float,
               seq_max_secs: int) -> pd.DataFrame:
    lignes = []
    seq_max_ns = seq_max_secs * NS
    i = 0
    while i < len(rejets) - 1:
        r1, r2 = rejets[i], rejets[i + 1]
        ecart_ns = pd.Timestamp(r2.fin).value - pd.Timestamp(r1.fin).value
        gap_points = (r1.B - r2.B) if sens == "buy" else (r2.B - r1.B)

        degrade = gap_points > 0                     # B2 plus faible que B1
        dans_le_temps = ecart_ns <= seq_max_ns
        dans_le_prix = 0 < gap_points <= gap_max_points

        if degrade and dans_le_temps and dans_le_prix:
            lignes.append({
                "t_rejet_1": r1.fin, "B1": r1.B, "vol_1": r1.volume,
                "t_rejet_2": r2.fin, "B2": r2.B, "vol_2": r2.volume,
                "gap_points": round(gap_points, 2),
                "ecart_secs": round(ecart_ns / NS, 1),
            })
            i += 2                # les deux evenements sont consommes
        else:
            i += 1
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------

def rapport(df: pd.DataFrame, sens: str):
    n = len(df)
    print("\n" + "=" * 64)
    print(f"SEQUENCES IDENTIFIEES ({sens}) : {n}")
    print("=" * 64)
    if not n:
        print("\nAucune sequence avec ces criteres.")
        return

    jours = df["t_rejet_2"].dt.date
    print(f"\nCouverture : {jours.min()} -> {jours.max()}  "
          f"({jours.nunique()} seances, {n/jours.nunique():.1f}/jour)")
    print(f"Ecart de prix (B1->B2)  : median {df['gap_points'].median():.2f} pts"
          f" | max {df['gap_points'].max():.2f} pts")
    print(f"Delai entre les 2 rejets : median {df['ecart_secs'].median():.0f} s"
          f" | max {df['ecart_secs'].max():.0f} s")
    print(f"Volume (info, non filtre) : evt.1 median {df['vol_1'].median():.0f}"
          f" | evt.2 median {df['vol_2'].median():.0f}")

    print("\n--- Liste complete ---")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))


def charger_plusieurs(chemins: list[str]) -> pd.DataFrame:
    """
    Charge et fusionne plusieurs fichiers de ticks.

    Si deux exports se chevauchent sur les memes dates (par ex. ticks_ES.csv
    et ticks_ES_full.csv), les lignes identiques (meme time/price/bid/ask/qty)
    sont dedupliquees pour ne pas compter un evenement deux fois.
    """
    morceaux = []
    for chemin in chemins:
        df = charger(chemin)
        print(f"  {chemin} : {len(df):,} ticks — "
              f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
        morceaux.append(df)

    full = pd.concat(morceaux, ignore_index=True)
    avant = len(full)
    full = full.drop_duplicates(subset=["time", "price", "bid", "ask", "qty"])
    doublons = avant - len(full)
    if doublons:
        print(f"  {doublons:,} lignes en double (chevauchement entre fichiers) retirees")
    return full.sort_values("time").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fichiers", nargs="+",
                    help="un ou plusieurs CSV de ticks, fusionnes automatiquement")
    ap.add_argument("--sens", choices=["buy", "sell"], default="buy",
                    help="buy = above-ask rejetes (acheteurs pieges, defaut)")
    ap.add_argument("--debut", default=SESSION_START)
    ap.add_argument("--fin", default=SESSION_END)
    ap.add_argument("--confirm", type=int, default=CONFIRM_SECS,
                    help="secondes sans depassement pour valider un rejet")
    ap.add_argument("--gap-max", type=float, default=GAP_MAX_POINTS,
                    help="ecart de prix max entre B1 et B2, en points")
    ap.add_argument("--seq-max", type=int, default=SEQ_MAX_SECS,
                    help="delai max entre les deux rejets, en secondes")
    ap.add_argument("--group-ms", type=int, default=GROUP_MS)
    args = ap.parse_args()

    print(f"Chargement de {len(args.fichiers)} fichier(s) ...")
    full = classifier(charger_plusieurs(args.fichiers))
    n = len(full)
    print(f"\n  Total fusionne : {n:,} ticks — "
          f"{full['time'].iloc[0]} -> {full['time'].iloc[-1]}")

    t = full["time"].dt.time
    d, f = pd.Timestamp(args.debut).time(), pd.Timestamp(args.fin).time()
    session = full[(t >= d) & (t <= f)]
    print(f"  Session {args.debut}-{args.fin} : {len(session):,} ticks "
          f"({len(session)/n:.1%})")

    colonne = "above" if args.sens == "buy" else "below"
    ev = grouper(session, session[colonne], args.sens, args.group_ms * 1_000_000)
    print(f"\n  {len(ev):,} evenements '{colonne}' regroupes "
          f"(fenetre {args.group_ms} ms)")
    if ev.empty:
        return

    px  = full["price"].to_numpy(dtype=np.float64)
    tns = full["time"].values.astype("datetime64[ns]").astype("int64")

    rejets = confirmer(ev, px, tns, args.sens, args.confirm * NS)
    print(f"  {len(rejets):,} rejets confirmes ({args.confirm}s sans depassement)")

    df = identifier(rejets, args.sens, args.gap_max, args.seq_max)
    print(f"  {len(df):,} sequences identifiees "
          f"(ecart <= {args.gap_max} pts, delai <= {args.seq_max}s)")

    rapport(df, args.sens)

    if len(df):
        out = Path(args.fichiers[0]).with_name(f"identifie_{args.sens}.csv")
        df.to_csv(out, index=False)
        print(f"\nExporte -> {out}")


if __name__ == "__main__":
    main()

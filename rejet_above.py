"""
Detection du double rejet a partir des VRAIS prints above-ask / below-bid.

Difference fondamentale avec rejet_sequence.py :

  rejet_sequence.py cherchait un amas de prints du meme cote traversant N
  niveaux en 1 seconde. Un prix qui derive de 3 ticks avec 30 lots satisfait
  ce critere : ce sont des dizaines de petits ordres qui prennent l'offre
  courante l'un apres l'autre. Jigsaw les imprime en At Ask, couleur normale.

  Ici on ne detecte QUE ce que Jigsaw colore differemment : une execution a un
  prix STRICTEMENT superieur au meilleur ask (ou inferieur au meilleur bid).
  Cela signifie qu'un ordre unique a consomme toute l'offre affichee et a
  continue au-dela, plus vite que le carnet ne s'est mis a jour.

  C'est un evenement, pas une accumulation.

Les executions contigues d'un meme ordre sont regroupees en un evenement,
comme le fait le tape reconstruit de Jigsaw.

Usage :
    python rejet_above.py ticks_ES_full.csv
    python rejet_above.py ticks_ES_full.csv --volume 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NS = 1_000_000_000
TICK = 0.25

SESSION_START = "13:30"
SESSION_END   = "14:45"

GROUP_MS      = 50     # executions plus rapprochees que ca = un seul ordre
MIN_VOLUME    = None   # plancher de volume de l'evenement, en lots
CONFIRM_SECS  = 30     # duree sans depassement pour valider un rejet
SEQ_MAX_SECS  = 600    # delai max entre les deux evenements
GAP_MIN_TICKS = 1
GAP_MAX_TICKS = 40
STOP_TICKS    = 1
MAX_RISQUE    = 3
HORIZON_MIN   = 60


# ---------------------------------------------------------------------------
# Chargement — on GARDE bid et ask, contrairement au loader du projet
# ---------------------------------------------------------------------------

def charger(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0,
                     names=["time", "price", "bid", "ask", "qty"])
    try:
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def classifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marque les executions selon les deux definitions possibles.

    'above' / 'below' : execution strictement hors du spread.

    'sweep' : definition de Jigsaw — "a sweep occurs when a buyer buys up all
    the contracts offered on the inside offer". On ne dispose pas des
    quantites du carnet, mais la consequence est observable : si toute
    l'offre est prise, le meilleur ask doit remonter juste apres. On marque
    donc les executions a l'offre suivies d'une remontee de l'ask.
    """
    spread_ok = df["ask"] > df["bid"]
    px, bid, ask = df["price"].values, df["bid"].values, df["ask"].values

    df["above"] = spread_ok & (df["price"] > df["ask"])
    df["below"] = spread_ok & (df["price"] < df["bid"])

    monte = np.append(ask[1:] > ask[:-1], False)     # l'offre s'eleve ensuite
    baisse = np.append(bid[1:] < bid[:-1], False)    # le bid s'efface ensuite
    df["sweep_buy"] = (px >= ask) & monte
    df["sweep_sell"] = (px <= bid) & baisse
    return df


# ---------------------------------------------------------------------------
# Regroupement facon tape reconstruit
# ---------------------------------------------------------------------------

def grouper(df: pd.DataFrame, masque, sens: str, group_ns: int):
    """
    Regroupe les executions hors spread contigues en evenements.

    Deux executions appartiennent au meme ordre si elles sont separees de
    moins de group_ns. C'est ce que Jigsaw affiche sur une seule ligne.

    Retourne un DataFrame : un evenement par ligne, avec son terminus B.
    """
    sub = df[masque]
    if sub.empty:
        return pd.DataFrame()

    t = sub["time"].values.astype("datetime64[ns]").astype("int64")
    # Nouvelle rupture des qu'un ecart depasse group_ns
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
# Rejet causal, appariement, simulation
# ---------------------------------------------------------------------------

def confirmer(ev, px, tns, sens, confirm_ns):
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
            rejets.append((r, r.B, t_conf))
    return rejets


def apparier(rejets, sens):
    paires, i = [], 0
    seq_max_ns = SEQ_MAX_SECS * NS
    while i < len(rejets) - 1:
        r1, B1, _ = rejets[i]
        r2, B2, t2 = rejets[i + 1]
        ecart = pd.Timestamp(r2.fin).value - pd.Timestamp(r1.fin).value
        gap = (B1 - B2) / TICK if sens == "buy" else (B2 - B1) / TICK
        if ecart <= seq_max_ns and GAP_MIN_TICKS <= gap <= GAP_MAX_TICKS:
            paires.append((r1, B1, r2, B2, t2, gap))
            i += 2
        else:
            i += 1
    return paires


def simuler(paires, px, tns, sens, stop_ticks, max_risque, horizon_min):
    short = sens == "buy"
    horizon_ns = horizon_min * 60 * NS
    lignes, trop_loin = [], 0

    for r1, B1, r2, B2, t_entree, gap in paires:
        lo = int(np.searchsorted(tns, t_entree, side="right"))
        hi = int(np.searchsorted(tns, t_entree + horizon_ns, side="right"))
        if hi <= lo:
            continue
        entree = float(px[lo - 1]) if lo > 0 else float(px[lo])
        stop = B2 + stop_ticks * TICK if short else B2 - stop_ticks * TICK
        risque = abs(stop - entree) / TICK
        if risque > max_risque:
            trop_loin += 1
            continue

        seg, segt = px[lo:hi], tns[lo:hi]
        touche = (seg >= stop) if short else (seg <= stop)
        if touche.any():
            j = int(np.argmax(touche))
            vie, viet, stoppe = seg[:j + 1], segt[:j + 1], True
        else:
            vie, viet, stoppe = seg, segt, False

        if short:
            mfe = (entree - float(vie.min())) / TICK
            mae = (float(vie.max()) - entree) / TICK
            j_mfe = int(np.argmin(vie))
        else:
            mfe = (float(vie.max()) - entree) / TICK
            mae = (entree - float(vie.min())) / TICK
            j_mfe = int(np.argmax(vie))

        lignes.append({
            "t_entree": pd.Timestamp(t_entree), "B1": B1, "B2": B2,
            "gap_ticks": round(gap, 1),
            "vol_1": r1.volume, "vol_2": r2.volume,
            "niv_1": r1.niveaux, "niv_2": r2.niveaux,
            "entree": entree, "stop": stop, "risque_ticks": round(risque, 1),
            "mfe_ticks": round(max(0.0, mfe), 1),
            "mae_ticks": round(max(0.0, mae), 1),
            "t_to_mfe_s": round((int(viet[j_mfe]) - t_entree) / NS, 1),
            "stoppe": stoppe,
        })

    if trop_loin:
        print(f"  {trop_loin} sequence(s) ecartee(s) : entree a plus de "
              f"{max_risque} ticks du stop")
    return pd.DataFrame(lignes)


def rapport(df):
    n = len(df)
    print("\n" + "=" * 64)
    print(f"DOUBLE REJET SUR ABOVE-ASK REELS : {n}")
    print("=" * 64)
    if not n:
        print("\nAucune sequence.")
        return

    jours = df["t_entree"].dt.date
    print(f"\nCouverture : {jours.min()} -> {jours.max()}  ({jours.nunique()} seances)")
    print(f"Risque median : {df['risque_ticks'].median():.1f} ticks")
    print(f"Taux de stop  : {df['stoppe'].mean():.0%}")
    print(f"Volume median de l'evenement d'entree : {df['vol_2'].median():.0f} lots")

    print("\n--- Distribution du MFE ---")
    bornes = [-0.1, 0, 2, 4, 8, 16, 40, 1e9]
    noms = ["0", "1-2", "3-4", "5-8", "9-16", "17-40", "40+"]
    b = pd.cut(df["mfe_ticks"], bins=bornes, labels=noms)
    cum = 0
    for nom in noms:
        c = int((b == nom).sum())
        cum += c
        print(f"  {nom:>6} ticks : {c:>5}  ({100*c/n:5.1f} %)   cumul {100*cum/n:5.1f} %")

    print("\n--- Paliers (esperance brute, AVANT frais ~1 tick) ---")
    risque = df["risque_ticks"].median()
    for cible in (2, 4, 8, 12, 20, 40):
        p = (df["mfe_ticks"] >= cible).mean()
        esp = p * cible - (1 - p) * risque
        # Erreur-type : sans elle un chiffre positif ne veut rien dire
        var = p * cible**2 + (1 - p) * risque**2 - esp**2
        se = (var / n) ** 0.5
        print(f"  >= {cible:>3} ticks : {p:6.1%}   esperance {esp:+6.2f} "
              f"+/- {se:.2f} ticks")

    print(f"\nMediane du temps jusqu'au MFE : {df['t_to_mfe_s'].median():.0f} s")

    print("\n--- Apercu ---")
    cols = ["t_entree", "B1", "B2", "gap_ticks", "vol_1", "vol_2",
            "risque_ticks", "mfe_ticks", "t_to_mfe_s", "stoppe"]
    pas = max(1, n // 15)
    with pd.option_context("display.width", 200):
        print(df[cols].iloc[::pas].head(15).to_string(index=False))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fichier")
    ap.add_argument("--sens", choices=["buy", "sell"], default="buy")
    ap.add_argument("--mode", choices=["sweep", "above"], default="sweep",
                    help="sweep = definition Jigsaw, toute l'offre consommee "
                         "(defaut) ; above = execution hors du spread")
    ap.add_argument("--debut", default=SESSION_START)
    ap.add_argument("--fin", default=SESSION_END)
    ap.add_argument("--volume", type=float, default=MIN_VOLUME,
                    help="volume minimum de l'evenement, en lots")
    ap.add_argument("--confirm", type=int, default=CONFIRM_SECS)
    ap.add_argument("--risque", type=float, default=MAX_RISQUE)
    ap.add_argument("--stop", type=float, default=STOP_TICKS)
    ap.add_argument("--horizon", type=int, default=HORIZON_MIN)
    ap.add_argument("--group-ms", type=int, default=GROUP_MS)
    args = ap.parse_args()

    print(f"Chargement de {args.fichier} ...")
    full = classifier(charger(args.fichier))
    n = len(full)
    print(f"  {n:,} ticks — {full['time'].iloc[0]} -> {full['time'].iloc[-1]}")

    # --- Les deux definitions, cote a cote -----------------------------------
    n_above, n_below = int(full["above"].sum()), int(full["below"].sum())
    n_sb, n_ss = int(full["sweep_buy"].sum()), int(full["sweep_sell"].sum())
    print(f"\n  Hors spread   — above ask : {n_above:,} ({100*n_above/n:.3f} %)"
          f" | below bid : {n_below:,} ({100*n_below/n:.3f} %)")
    print(f"  Offre nettoyee — buy : {n_sb:,} ({100*n_sb/n:.3f} %)"
          f" | sell : {n_ss:,} ({100*n_ss/n:.3f} %)")

    if args.mode == "above" and n_above + n_below == 0:
        print("\n  BLOQUANT : aucune execution hors du spread dans ce fichier.")
        print("  L'export enregistre probablement le quote APRES le trade.")
        print("  Relance avec --mode sweep, qui ne depend pas de cette condition.")
        sys.exit(1)
    if args.mode == "sweep" and n_sb + n_ss == 0:
        print("\n  BLOQUANT : le quote ne bouge jamais apres une execution.")
        print("  Les colonnes bid/ask de cet export ne sont pas exploitables.")
        sys.exit(1)

    t = full["time"].dt.time
    d, f = pd.Timestamp(args.debut).time(), pd.Timestamp(args.fin).time()
    session = full[(t >= d) & (t <= f)]
    print(f"\n  Session {args.debut}-{args.fin} : {len(session):,} ticks "
          f"({len(session)/n:.1%})")

    if args.mode == "sweep":
        colonne = "sweep_buy" if args.sens == "buy" else "sweep_sell"
    else:
        colonne = "above" if args.sens == "buy" else "below"
    ev = grouper(session, session[colonne], args.sens, args.group_ms * 1_000_000)
    print(f"\n  {len(ev):,} evenements '{args.mode}' regroupes "
          f"(fenetre {args.group_ms} ms)")
    if ev.empty:
        return
    print(f"  Volume : median {ev['volume'].median():.0f} lots | "
          f"90e centile {ev['volume'].quantile(0.9):.0f} | max {ev['volume'].max():.0f}")
    print(f"  Niveaux traverses : median {ev['niveaux'].median():.0f} | "
          f"max {ev['niveaux'].max()}")

    if args.volume:
        avant = len(ev)
        ev = ev[ev["volume"] >= args.volume]
        print(f"  Apres plancher {args.volume:.0f} lots : {len(ev):,} / {avant:,}")
        if ev.empty:
            return

    px  = full["price"].to_numpy(dtype=np.float64)
    tns = full["time"].values.astype("datetime64[ns]").astype("int64")

    rejets = confirmer(ev, px, tns, args.sens, args.confirm * NS)
    print(f"  {len(rejets):,} rejets confirmes ({args.confirm}s sans depassement)")

    paires = apparier(rejets, args.sens)
    print(f"  {len(paires):,} sequences de double rejet")

    df = simuler(paires, px, tns, args.sens, args.stop, args.risque, args.horizon)
    rapport(df)

    if len(df):
        out = Path(args.fichier).with_name(f"above_rejets_{args.sens}.csv")
        df.to_csv(out, index=False)
        print(f"\nExporte -> {out}")


if __name__ == "__main__":
    main()

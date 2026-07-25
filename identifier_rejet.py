"""
Identification du pattern de double rejet + deux analyses complementaires :

  A. Verification du "chemin du retour" : entre la confirmation du 2e rejet
     et l'issue (invalidation ou horizon), y a-t-il un rejet CONFIRME du
     cote OPPOSE (une rejection vendeuse qui vient contester la baisse
     attendue apres un double rejet acheteur, et inversement) ?

  B. Backtest approximatif : entree au marche juste apres confirmation du
     2e rejet, stop juste au-dela de B2, cible a --target points. Resultats
     bruts, a considerer comme un premier ordre de grandeur.

Pattern de base (inchange) :
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
affiche pour information.

Usage :
    python identifier_rejet.py ticks_ES_full.csv
    python identifier_rejet.py ticks_ES_full.csv --sens sell --confirm 5
    python identifier_rejet.py ticks_ES_full.csv --target 1.5 --stop-buffer 0.5
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
SEQ_MIN_SECS   = 5      # plancher : en dessous, probablement le meme sursaut
GAP_MAX_POINTS = 3      # ecart max entre B1 et B2, en points d'indice

TARGET_POINTS  = 1.0    # cible du backtest, en points
STOP_BUFFER    = 0.25   # marge du stop au-dela de B2, en points (1 tick)
COUT_ALLER_RETOUR = 0.25  # cout estime d'un aller-retour, en points (info)


# ---------------------------------------------------------------------------
# Chargement + classification
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
               seq_max_secs: int, seq_min_secs: float = 0) -> pd.DataFrame:
    lignes = []
    seq_max_ns = seq_max_secs * NS
    seq_min_ns = seq_min_secs * NS
    i = 0
    while i < len(rejets) - 1:
        r1, r2 = rejets[i], rejets[i + 1]
        ecart_ns = pd.Timestamp(r2.fin).value - pd.Timestamp(r1.fin).value
        gap_points = (r1.B - r2.B) if sens == "buy" else (r2.B - r1.B)

        degrade = gap_points > 0                     # B2 plus faible que B1
        # sous seq_min_secs : probablement le meme sursaut fragmente en deux
        # evenements, pas deux tentatives distinctes
        dans_le_temps = seq_min_ns <= ecart_ns <= seq_max_ns
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
# Mesure de la suite — combien de points avant que B2 finisse par ceder ?
# Aucune decision de trade ici (pas de stop, pas d'entree) : on mesure juste
# la duree de vie du niveau, comme le fait enrich.py pour le modele A/B.
# ---------------------------------------------------------------------------

def mesurer_suite(df: pd.DataFrame, px: np.ndarray, tns: np.ndarray,
                   sens: str, confirm_secs: int, horizon_secs: int = 3600) -> pd.DataFrame:
    """
    A partir du moment ou le 2e rejet est confirme, mesure :
      - ext_max_pts  : plus grand ecart favorable atteint (B2 tient, le prix
                        s'en eloigne) avant que B2 soit repasse dans l'autre sens
      - t_ext_max_s  : temps ecoule jusqu'a ce point
      - invalide     : est-ce que B2 a fini par ceder dans l'horizon ?
      - t_invalide_s : temps ecoule jusqu'a la cession, si applicable
    """
    horizon_ns = horizon_secs * NS
    confirm_ns = confirm_secs * NS
    ext_max, t_ext_max, invalide, t_invalide = [], [], [], []

    for r in df.itertuples():
        origin_ns = pd.Timestamp(r.t_rejet_2).value + confirm_ns
        cap_ns = origin_ns + horizon_ns
        lo = int(np.searchsorted(tns, origin_ns, side="right"))
        hi = int(np.searchsorted(tns, cap_ns, side="right"))
        if hi <= lo:
            ext_max.append(np.nan); t_ext_max.append(np.nan)
            invalide.append(False); t_invalide.append(np.nan)
            continue

        seg, segt = px[lo:hi], tns[lo:hi]
        if sens == "buy":               # rejet acheteur -> favorable = prix baisse
            favorable = r.B2 - seg
            casse = seg > r.B2
        else:                           # rejet vendeur -> favorable = prix monte
            favorable = seg - r.B2
            casse = seg < r.B2

        if casse.any():
            j = int(np.argmax(casse))
            pre = favorable[:j + 1]
            invalide.append(True)
            t_invalide.append(round((int(segt[j]) - origin_ns) / NS, 1))
        else:
            pre = favorable
            invalide.append(False)
            t_invalide.append(np.nan)

        jm = int(np.argmax(pre))
        ext_max.append(round(max(0.0, float(pre[jm])), 2))
        t_ext_max.append(round((int(segt[jm]) - origin_ns) / NS, 1))

    df = df.copy()
    df["ext_max_pts"] = ext_max
    df["t_ext_max_s"] = t_ext_max
    df["invalide"] = invalide
    df["t_invalide_s"] = t_invalide
    return df


# ---------------------------------------------------------------------------
# A. Chemin du retour — un rejet confirme du cote OPPOSE apparait-il avant
#    l'issue (invalidation ou horizon) ? Purement descriptif : compte le
#    nombre de rejets opposes dans la fenetre et le delai jusqu'au premier.
# ---------------------------------------------------------------------------

def verifier_chemin_oppose(df: pd.DataFrame, rejets_opposes: list,
                            confirm_secs: int, horizon_secs: int) -> pd.DataFrame:
    confirm_ns = confirm_secs * NS
    horizon_ns = horizon_secs * NS

    fins_opp = np.array(sorted(pd.Timestamp(r.fin).value for r in rejets_opposes),
                         dtype="int64")

    n_contra, t_premier = [], []
    for r in df.itertuples():
        origin_ns = pd.Timestamp(r.t_rejet_2).value + confirm_ns
        if bool(r.invalide):
            fin_fenetre_ns = origin_ns + int(round(r.t_invalide_s * NS))
        else:
            fin_fenetre_ns = origin_ns + horizon_ns

        if fins_opp.size:
            lo = int(np.searchsorted(fins_opp, origin_ns, side="right"))
            hi = int(np.searchsorted(fins_opp, fin_fenetre_ns, side="right"))
        else:
            lo = hi = 0
        n = hi - lo
        n_contra.append(n)
        t_premier.append(round((fins_opp[lo] - origin_ns) / NS, 1) if n > 0 else np.nan)

    df = df.copy()
    df["contra_rejets"] = n_contra
    df["t_premier_contra_s"] = t_premier
    return df


# ---------------------------------------------------------------------------
# B. Backtest approximatif — entree au marche, stop au-dela de B2, cible a
#    --target points. Premier des deux touche = issue du trade.
# ---------------------------------------------------------------------------

def simuler_trade(df: pd.DataFrame, px: np.ndarray, tns: np.ndarray, sens: str,
                   confirm_secs: int, target_points: float, stop_buffer: float,
                   horizon_secs: int) -> pd.DataFrame:
    confirm_ns = confirm_secs * NS
    horizon_ns = horizon_secs * NS

    pnl, issue, t_issue, entree_l = [], [], [], []

    for r in df.itertuples():
        origin_ns = pd.Timestamp(r.t_rejet_2).value + confirm_ns
        cap_ns = origin_ns + horizon_ns
        lo = int(np.searchsorted(tns, origin_ns, side="right"))
        hi = int(np.searchsorted(tns, cap_ns, side="right"))
        if hi <= lo:
            pnl.append(np.nan); issue.append("pas_de_donnees")
            t_issue.append(np.nan); entree_l.append(np.nan)
            continue

        entree = float(px[lo])
        seg, segt = px[lo:hi], tns[lo:hi]

        if sens == "buy":                       # double rejet acheteur -> on VEND
            stop = r.B2 + stop_buffer
            cible = entree - target_points
            touche_stop = seg >= stop
            touche_cible = seg <= cible
        else:                                   # double rejet vendeur -> on ACHETE
            stop = r.B2 - stop_buffer
            cible = entree + target_points
            touche_stop = seg <= stop
            touche_cible = seg >= cible

        i_stop = int(np.argmax(touche_stop)) if touche_stop.any() else None
        i_cible = int(np.argmax(touche_cible)) if touche_cible.any() else None

        entree_l.append(round(entree, 2))
        if i_stop is None and i_cible is None:
            pnl.append(np.nan); issue.append("timeout")
            t_issue.append(np.nan)
        elif i_cible is not None and (i_stop is None or i_cible <= i_stop):
            pnl.append(target_points); issue.append("gagnant")
            t_issue.append(round((int(segt[i_cible]) - origin_ns) / NS, 1))
        else:
            perte = -abs(entree - stop)
            pnl.append(round(perte, 2)); issue.append("perdant")
            t_issue.append(round((int(segt[i_stop]) - origin_ns) / NS, 1))

    df = df.copy()
    df["entree"] = entree_l
    df["pnl_pts"] = pnl
    df["issue"] = issue
    df["t_issue_s"] = t_issue
    return df


# ---------------------------------------------------------------------------

def rapport(df: pd.DataFrame, sens: str, target_points: float, cout: float):
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

    if "ext_max_pts" in df.columns:
        print("\n--- Combien de points avant que B2 finisse par ceder ---")
        print("(mesure pure : pas de stop, pas d'entree — juste la duree de vie du niveau)")
        bornes = [-0.01, 0, 1, 2, 3, 5, 8, 1e9]
        noms = ["0", "0-1", "1-2", "2-3", "3-5", "5-8", "8+"]
        b = pd.cut(df["ext_max_pts"], bins=bornes, labels=noms)
        cum = 0
        for nom in noms:
            c = int((b == nom).sum())
            cum += c
            print(f"  {nom:>4} pts : {c:>4}  ({100*c/n:5.1f} %)   cumul {100*cum/n:5.1f} %")
        print(f"  Mediane du temps jusqu'au max : {df['t_ext_max_s'].median():.0f} s")
        print(f"  B2 finit par ceder (dans l'horizon) : {df['invalide'].mean():.0%}")
        if df["invalide"].any():
            print(f"  Temps median avant que ca cede      : "
                  f"{df.loc[df['invalide'], 't_invalide_s'].median():.0f} s")

    if "contra_rejets" in df.columns:
        print("\n--- A. Chemin du retour : rejet confirme du cote OPPOSE ? ---")
        cote_opp = "vendeuse" if sens == "buy" else "acheteuse"
        print(f"(entre confirmation du 2e rejet et l'issue, presence d'une rejection {cote_opp})")
        propre = int((df["contra_rejets"] == 0).sum())
        conteste = n - propre
        print(f"  Chemin propre (0 rejet oppose)   : {propre:>4}  ({100*propre/n:5.1f} %)")
        print(f"  Chemin conteste (>=1 rejet oppose): {conteste:>4}  ({100*conteste/n:5.1f} %)")
        if propre and conteste:
            ext_propre = df.loc[df["contra_rejets"] == 0, "ext_max_pts"].median()
            ext_conteste = df.loc[df["contra_rejets"] > 0, "ext_max_pts"].median()
            inv_propre = df.loc[df["contra_rejets"] == 0, "invalide"].mean()
            inv_conteste = df.loc[df["contra_rejets"] > 0, "invalide"].mean()
            print(f"  Extension mediane si propre   : {ext_propre:.2f} pts"
                  f"  | B2 cede : {inv_propre:.0%}")
            print(f"  Extension mediane si conteste : {ext_conteste:.2f} pts"
                  f"  | B2 cede : {inv_conteste:.0%}")
            avec_contra = df[df["contra_rejets"] > 0]
            delai_median = avec_contra["t_premier_contra_s"].median()
            print(f"  Delai median avant le premier rejet oppose : {delai_median:.0f} s")

    if "pnl_pts" in df.columns:
        print(f"\n--- B. Backtest approximatif (marche, cible {target_points} pts, "
              f"stop = B2 +/- buffer) ---")
        print("(ordre de grandeur seulement — pas de gestion de position, un trade a la fois suppose)")
        vc = df["issue"].value_counts()
        for k in ("gagnant", "perdant", "timeout", "pas_de_donnees"):
            c = int(vc.get(k, 0))
            print(f"  {k:<14}: {c:>4}  ({100*c/n:5.1f} %)")
        valides = df["pnl_pts"].dropna()
        if len(valides):
            esp = valides.mean()
            se = valides.std(ddof=1) / np.sqrt(len(valides))
            esp_net = esp - cout
            print(f"  Esperance brute  : {esp:+.3f} pts  (+/- {se:.3f} SE, n={len(valides)})")
            print(f"  Esperance nette  : {esp_net:+.3f} pts  (cout aller-retour estime {cout} pt)")
            print(f"  Perte moyenne    : {valides[valides < 0].mean():.2f} pts"
                  if (valides < 0).any() else "  Perte moyenne    : n/a")

    print("\n--- Distribution horaire (heure du 2e rejet) ---")
    heures = df["t_rejet_2"].dt.hour
    for h in sorted(heures.unique()):
        c = int((heures == h).sum())
        barre = "#" * int(40 * c / heures.value_counts().max())
        print(f"  {h:02d}h : {c:>4}  {barre}")

    print("\n--- Liste complete ---")
    with pd.option_context("display.max_rows", None, "display.width", 220):
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
    ap.add_argument("--seq-min", type=float, default=SEQ_MIN_SECS,
                    help="delai min entre les deux rejets, en secondes "
                         "(exclut les fragments d'un meme sursaut)")
    ap.add_argument("--group-ms", type=int, default=GROUP_MS)
    ap.add_argument("--horizon", type=int, default=3600,
                    help="duree max de suivi apres confirmation, en secondes "
                         "(defaut 3600 = 1h)")
    ap.add_argument("--target", type=float, default=TARGET_POINTS,
                    help="cible du backtest, en points (defaut 1.0)")
    ap.add_argument("--stop-buffer", type=float, default=STOP_BUFFER,
                    help="marge du stop au-dela de B2, en points (defaut 0.25 = 1 tick)")
    ap.add_argument("--cout", type=float, default=COUT_ALLER_RETOUR,
                    help="cout estime d'un aller-retour, en points, "
                         "utilise seulement pour l'esperance nette (info)")
    ap.add_argument("--sans-chemin-oppose", action="store_true",
                    help="desactive la verification A (plus rapide)")
    ap.add_argument("--sans-backtest", action="store_true",
                    help="desactive le backtest B")
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

    px  = full["price"].to_numpy(dtype=np.float64)
    tns = full["time"].values.astype("datetime64[ns]").astype("int64")
    confirm_ns = args.confirm * NS

    ev_above = grouper(session, session["above"], "buy", args.group_ms * 1_000_000)
    ev_below = grouper(session, session["below"], "sell", args.group_ms * 1_000_000)
    print(f"\n  {len(ev_above):,} evenements 'above' regroupes / "
          f"{len(ev_below):,} evenements 'below' regroupes (fenetre {args.group_ms} ms)")

    rejets_above = confirmer(ev_above, px, tns, "buy", confirm_ns) if len(ev_above) else []
    rejets_below = confirmer(ev_below, px, tns, "sell", confirm_ns) if len(ev_below) else []
    print(f"  {len(rejets_above):,} rejets 'above' confirmes / "
          f"{len(rejets_below):,} rejets 'below' confirmes ({args.confirm}s sans depassement)")

    if args.sens == "buy":
        rejets_principal, rejets_oppose = rejets_above, rejets_below
    else:
        rejets_principal, rejets_oppose = rejets_below, rejets_above

    df = identifier(rejets_principal, args.sens, args.gap_max, args.seq_max, args.seq_min)
    print(f"  {len(df):,} sequences identifiees "
          f"(ecart <= {args.gap_max} pts, delai {args.seq_min}-{args.seq_max}s)")

    if len(df):
        df = mesurer_suite(df, px, tns, args.sens, args.confirm, args.horizon)
        if not args.sans_chemin_oppose:
            df = verifier_chemin_oppose(df, rejets_oppose, args.confirm, args.horizon)
        if not args.sans_backtest:
            df = simuler_trade(df, px, tns, args.sens, args.confirm,
                                args.target, args.stop_buffer, args.horizon)

    rapport(df, args.sens, args.target, args.cout)

    if len(df):
        out = Path(args.fichiers[0]).with_name(f"identifie_{args.sens}.csv")
        df.to_csv(out, index=False)
        print(f"\nExporte -> {out}")


if __name__ == "__main__":
    main()

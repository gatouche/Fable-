"""
Balayage de parametres de backtest sur le pattern double rejet.
Charge les ticks UNE SEULE FOIS, identifie les sequences (avec chemin oppose),
puis boucle sur une grille de parametres.

  Grille A/B : entree immediate apres confirmation du pattern
               x {tous / propre / conteste} x stop_buffer

  Variante C  : entree apres la confirmation du premier rejet oppose
                (le "signal" naturel en temps reel, sans filtre retrospectif)
                x fenetre_attente x stop_buffer

Usage :
    py grid_bt.py ticks_ES.csv ticks_ES_full.csv ticks_ES1.csv ticks_ES2.csv
    py grid_bt.py ticks_ES.csv ... --sens sell
    py grid_bt.py ticks_ES.csv ... --stops 0.25 0.5 0.75 1.0 --target 1.5
    py grid_bt.py ticks_ES.csv ... --fenetres 20 40 60 120
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from identifier_rejet import (
    NS, GROUP_MS, CONFIRM_SECS, SEQ_MAX_SECS, SEQ_MIN_SECS,
    GAP_MAX_POINTS, TARGET_POINTS,
    charger_plusieurs, classifier, grouper, confirmer,
    identifier, mesurer_suite, verifier_chemin_oppose, simuler_trade,
)


def stats_bt(df: pd.DataFrame, target: float, cout: float) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "winrate": np.nan, "esp_brut": np.nan,
                "se": np.nan, "esp_net": np.nan, "perte_moy": np.nan}
    vc = df["issue"].value_counts()
    valides = df["pnl_pts"].dropna()
    esp = valides.mean() if len(valides) else np.nan
    se = (valides.std(ddof=1) / np.sqrt(len(valides))
          if len(valides) > 1 else np.nan)
    return {
        "n": n,
        "winrate": 100 * int(vc.get("gagnant", 0)) / n,
        "esp_brut": esp,
        "se": se,
        "esp_net": esp - cout if not np.isnan(esp) else np.nan,
        "perte_moy": valides[valides < 0].mean() if (valides < 0).any() else np.nan,
    }


def simuler_trade_apres_contra(df_all: pd.DataFrame, rejets_opposes: list,
                                px: np.ndarray, tns: np.ndarray, sens: str,
                                confirm_secs: int, target_points: float,
                                stop_buffer: float, horizon_secs: int,
                                fenetre_attente_secs: int) -> pd.DataFrame:
    """
    Variante C - entree apres contra-rejet :
      1. Pattern confirme a T.
      2. Attend le premier rejet confirme du cote oppose dans la fenetre.
      3. Entre au marche au premier tick apres cette confirmation.
      4. Stop toujours au-dela de B2 (niveau original du pattern).
      5. Si aucun signal dans fenetre_attente_secs -> pas_de_signal (trade saute).

    En temps reel cette approche ne necessite aucune information future :
    on attend un evenement observable sur le tape avant d'entrer.
    """
    confirm_ns  = confirm_secs * NS
    horizon_ns  = horizon_secs * NS
    attente_ns  = fenetre_attente_secs * NS

    # Moment ou chaque rejet oppose EST confirme (fin de l'evenement + confirm_ns)
    fins_opp = np.array(
        sorted(pd.Timestamp(r.fin).value + confirm_ns for r in rejets_opposes),
        dtype="int64",
    ) if rejets_opposes else np.array([], dtype="int64")

    pnl, issue, t_issue, entree_l, t_signal_l = [], [], [], [], []

    for r in df_all.itertuples():
        origin_ns  = pd.Timestamp(r.t_rejet_2).value + confirm_ns
        fenetre_ns = origin_ns + attente_ns
        cap_ns     = origin_ns + horizon_ns

        # Premier rejet oppose confirme dans la fenetre d'attente
        t_signal_ns = None
        if fins_opp.size:
            lo = int(np.searchsorted(fins_opp, origin_ns, side="right"))
            if lo < fins_opp.size and fins_opp[lo] <= fenetre_ns:
                t_signal_ns = int(fins_opp[lo])

        if t_signal_ns is None:
            pnl.append(np.nan);  issue.append("pas_de_signal")
            t_issue.append(np.nan); entree_l.append(np.nan)
            t_signal_l.append(np.nan)
            continue

        t_signal_l.append(round((t_signal_ns - origin_ns) / NS, 1))

        lo_e = int(np.searchsorted(tns, t_signal_ns, side="right"))
        hi_e = int(np.searchsorted(tns, cap_ns,       side="right"))

        if hi_e <= lo_e:
            pnl.append(np.nan); issue.append("pas_de_donnees")
            t_issue.append(np.nan); entree_l.append(np.nan)
            continue

        entree = float(px[lo_e])
        seg, segt = px[lo_e:hi_e], tns[lo_e:hi_e]

        if sens == "buy":           # double rejet acheteur -> on VEND
            stop   = r.B2 + stop_buffer
            cible  = entree - target_points
            touche_stop  = seg >= stop
            touche_cible = seg <= cible
        else:                       # double rejet vendeur -> on ACHETE
            stop   = r.B2 - stop_buffer
            cible  = entree + target_points
            touche_stop  = seg <= stop
            touche_cible = seg >= cible

        i_stop  = int(np.argmax(touche_stop))  if touche_stop.any()  else None
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

    out = df_all.copy()
    out["t_signal_s"] = t_signal_l
    out["entree"]     = entree_l
    out["pnl_pts"]    = pnl
    out["issue"]      = issue
    out["t_issue_s"]  = t_issue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fichiers", nargs="+")
    ap.add_argument("--sens", choices=["buy", "sell"], default="buy")
    ap.add_argument("--debut", default="00:00")
    ap.add_argument("--fin", default="23:59")
    ap.add_argument("--confirm", type=int, default=CONFIRM_SECS)
    ap.add_argument("--gap-max", type=float, default=GAP_MAX_POINTS)
    ap.add_argument("--seq-max", type=int, default=SEQ_MAX_SECS)
    ap.add_argument("--seq-min", type=float, default=SEQ_MIN_SECS)
    ap.add_argument("--group-ms", type=int, default=GROUP_MS)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--target", type=float, default=TARGET_POINTS,
                    help="cible du backtest en points (defaut 1.0)")
    ap.add_argument("--cout", type=float, default=0.25,
                    help="cout aller-retour estime en points (defaut 0.25)")
    ap.add_argument("--stops", nargs="+", type=float,
                    default=[0.25, 0.50, 0.75, 1.00],
                    help="stop_buffer a tester (defaut: 0.25 0.50 0.75 1.00)")
    ap.add_argument("--fenetres", nargs="+", type=int,
                    default=[20, 40, 60, 120],
                    help="fenetres d'attente du contra-rejet en secondes "
                         "(variante C, defaut: 20 40 60 120)")
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Chargement et identification — une seule fois                        #
    # ------------------------------------------------------------------ #
    print(f"Chargement de {len(args.fichiers)} fichier(s) ...")
    full = classifier(charger_plusieurs(args.fichiers))
    n_total = len(full)
    print(f"  Total : {n_total:,} ticks — "
          f"{full['time'].iloc[0]} -> {full['time'].iloc[-1]}")

    t = full["time"].dt.time
    d, f = pd.Timestamp(args.debut).time(), pd.Timestamp(args.fin).time()
    session = full[(t >= d) & (t <= f)]
    print(f"  Session {args.debut}-{args.fin} : {len(session):,} ticks "
          f"({len(session)/n_total:.1%})")

    px  = full["price"].to_numpy(dtype=np.float64)
    tns = full["time"].values.astype("datetime64[ns]").astype("int64")
    confirm_ns = args.confirm * NS

    ev_above = grouper(session, session["above"], "buy",  args.group_ms * 1_000_000)
    ev_below = grouper(session, session["below"], "sell", args.group_ms * 1_000_000)
    print(f"\n  {len(ev_above):,} evenements 'above' / "
          f"{len(ev_below):,} evenements 'below' (fenetre {args.group_ms} ms)")

    rejets_above = confirmer(ev_above, px, tns, "buy",  confirm_ns) if len(ev_above) else []
    rejets_below = confirmer(ev_below, px, tns, "sell", confirm_ns) if len(ev_below) else []
    print(f"  {len(rejets_above):,} rejets 'above' / "
          f"{len(rejets_below):,} rejets 'below' ({args.confirm}s)")

    if args.sens == "buy":
        rejets_principal, rejets_oppose = rejets_above, rejets_below
    else:
        rejets_principal, rejets_oppose = rejets_below, rejets_above

    df_base = identifier(rejets_principal, args.sens,
                         args.gap_max, args.seq_max, args.seq_min)
    n_seq = len(df_base)
    print(f"  {n_seq:,} sequences identifiees "
          f"(ecart <= {args.gap_max} pts, delai {args.seq_min}-{args.seq_max}s)")

    if not n_seq:
        print("Aucune sequence — arret.")
        return

    df_base = mesurer_suite(df_base, px, tns, args.sens, args.confirm, args.horizon)
    df_base = verifier_chemin_oppose(df_base, rejets_oppose, args.confirm, args.horizon)

    df_propre   = df_base[df_base["contra_rejets"] == 0].reset_index(drop=True)
    df_conteste = df_base[df_base["contra_rejets"] >  0].reset_index(drop=True)
    print(f"\n  Chemin propre    : {len(df_propre):,} ({100*len(df_propre)/n_seq:.1f} %)")
    print(f"  Chemin conteste  : {len(df_conteste):,} ({100*len(df_conteste)/n_seq:.1f} %)")

    W = 96

    # ------------------------------------------------------------------ #
    # Grille A/B : entree immediate                                         #
    # ------------------------------------------------------------------ #
    filtres = [
        ("Tous",      df_base),
        ("Propre",    df_propre),
        ("Conteste",  df_conteste),
    ]

    print("\n" + "=" * W)
    print(f"  GRILLE A/B — entree immediate  |  {args.sens.upper()}  |  "
          f"cible {args.target} pt  |  cout {args.cout} pt  |  horizon {args.horizon} s")
    print("=" * W)
    print(f"  {'Stop':>6}  {'Filtre':<10}  {'n':>5}  "
          f"{'Win%':>5}  {'Esp.brut':>9}  {'±SE':>6}  "
          f"{'Esp.net':>8}  {'Perte moy':>9}")
    print("  " + "-" * (W - 2))

    for stop_buf in args.stops:
        for label, df_f in filtres:
            if not len(df_f):
                print(f"  {stop_buf:>6.2f}  {label:<10}  {'(vide)':>5}")
                continue
            df_sim = simuler_trade(df_f, px, tns, args.sens, args.confirm,
                                   args.target, stop_buf, args.horizon)
            s = stats_bt(df_sim, args.target, args.cout)
            pm = f"{s['perte_moy']:>9.2f}" if not np.isnan(s['perte_moy']) else f"{'n/a':>9}"
            print(f"  {stop_buf:>6.2f}  {label:<10}  {s['n']:>5}  "
                  f"{s['winrate']:>5.1f}%  "
                  f"{s['esp_brut']:>+9.3f}  "
                  f"{s['se']:>6.3f}  "
                  f"{s['esp_net']:>+8.3f}  {pm}")
        print()

    print("=" * W)

    # ------------------------------------------------------------------ #
    # Variante C : entree apres contra-rejet                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * W}")
    print(f"  VARIANTE C — entree apres contra-rejet  |  {args.sens.upper()}  |  "
          f"cible {args.target} pt  |  cout {args.cout} pt")
    print(f"  (attend la confirmation du 1er rejet oppose sur le tape, "
          f"puis entre au marche — aucune information future utilisee)")
    print("=" * W)
    print(f"  {'Fen.(s)':>7}  {'Stop':>6}  {'n_tot':>5}  {'n_trade':>7}  "
          f"{'couv%':>6}  {'Win%':>5}  {'Esp.brut':>9}  {'±SE':>6}  "
          f"{'Esp.net':>8}  {'Perte moy':>9}")
    print("  " + "-" * (W - 2))

    for fen in args.fenetres:
        for stop_buf in args.stops:
            df_sim = simuler_trade_apres_contra(
                df_base, rejets_oppose, px, tns, args.sens, args.confirm,
                args.target, stop_buf, args.horizon, fen,
            )
            df_trades = df_sim[df_sim["issue"] != "pas_de_signal"].reset_index(drop=True)
            n_trade = len(df_trades)
            couv = 100 * n_trade / n_seq if n_seq else 0.0
            s = stats_bt(df_trades, args.target, args.cout)
            pm = f"{s['perte_moy']:>9.2f}" if not np.isnan(s['perte_moy']) else f"{'n/a':>9}"
            print(f"  {fen:>7}  {stop_buf:>6.2f}  {n_seq:>5}  {n_trade:>7}  "
                  f"{couv:>6.1f}%  {s['winrate']:>5.1f}%  "
                  f"{s['esp_brut']:>+9.3f}  {s['se']:>6.3f}  "
                  f"{s['esp_net']:>+8.3f}  {pm}")
        print()

    print("=" * W)
    print("(ordre de grandeur — un trade a la fois suppose, "
          "pas de gestion de position, pas de slippage)")


if __name__ == "__main__":
    main()

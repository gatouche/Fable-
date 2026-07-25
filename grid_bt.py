"""
Balayage de parametres de backtest sur le pattern double rejet.
Charge les ticks UNE SEULE FOIS, identifie les sequences (avec chemin oppose),
puis boucle sur une grille de stop_buffer x {tous / propre / conteste}.

Usage :
    py grid_bt.py ticks_ES.csv ticks_ES_full.csv ticks_ES1.csv ticks_ES2.csv
    py grid_bt.py ticks_ES.csv ... --sens sell
    py grid_bt.py ticks_ES.csv ... --stops 0.25 0.5 0.75 1.0 --target 1.5
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

    # ------------------------------------------------------------------ #
    # Grille : stop_buffer x filtre                                        #
    # ------------------------------------------------------------------ #
    filtres = [
        ("Tous",            df_base),
        ("Propre",          df_propre),
        ("Conteste",        df_conteste),
    ]

    W = 92
    print("\n" + "=" * W)
    print(f"  GRILLE BACKTEST  |  {args.sens.upper()}  |  "
          f"cible {args.target} pt  |  cout {args.cout} pt  |  "
          f"horizon {args.horizon} s")
    print("=" * W)
    print(f"  {'Stop buf':>8}  {'Filtre':<12}  {'n':>5}  "
          f"{'Win%':>5}  {'Esp.brut':>9}  {'±SE':>6}  "
          f"{'Esp.net':>8}  {'Perte moy':>9}")
    print("  " + "-" * (W - 2))

    for stop_buf in args.stops:
        for label, df_f in filtres:
            if not len(df_f):
                print(f"  {stop_buf:>8.2f}  {label:<12}  {'(vide)':>5}")
                continue
            df_sim = simuler_trade(df_f, px, tns, args.sens, args.confirm,
                                   args.target, stop_buf, args.horizon)
            s = stats_bt(df_sim, args.target, args.cout)
            perte_s = f"{s['perte_moy']:>9.2f}" if not np.isnan(s['perte_moy']) else f"{'n/a':>9}"
            print(f"  {stop_buf:>8.2f}  {label:<12}  {s['n']:>5}  "
                  f"{s['winrate']:>5.1f}%  "
                  f"{s['esp_brut']:>+9.3f}  "
                  f"{s['se']:>6.3f}  "
                  f"{s['esp_net']:>+8.3f}  "
                  f"{perte_s}")
        print()

    print("=" * W)
    print("(ordre de grandeur — un trade a la fois suppose, "
          "pas de gestion de position, pas de slippage)")


if __name__ == "__main__":
    main()

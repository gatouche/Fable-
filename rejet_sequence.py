"""
Detection de la sequence de double rejet sur la tape ES.

Pattern vise (decrit par l'utilisateur, premiere heure de seance) :

    sweep 1 (buy) monte jusqu'a B1  ->  le prix ne repasse jamais au-dessus
    sweep 2 (buy) monte jusqu'a B2 < B1  ->  le prix ne repasse pas non plus
    => les acheteurs agressifs sont piegres deux fois, a un niveau decroissant
    => short, stop 1 tick au-dessus de B2

Reutilise le detecteur existant (sweep.detect_tape_sweeps) avec des seuils
abaisses : le pattern porte sur des sweeps de 2 a 6 ticks, pas sur les gros
sweeps de 11+ niveaux du pipeline d'origine.

DIFFERENCE CLE avec enrich_sweeps : ici tout est CAUSAL. Un rejet est confirme
apres CONFIRM_SECS sans depassement, pas en regardant jusqu'a la fin du RTH.
On peut donc entrer dessus sans biais de look-ahead.

Usage :
    python rejet_sequence.py ticks_ES_full.csv
    python rejet_sequence.py ticks_ES_full.csv --sens sell
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sweep import detect_tape_sweeps

NS = 1_000_000_000

# ---------------------------------------------------------------------------
# Parametres — tout ce qui se regle est ici
# ---------------------------------------------------------------------------

TICK = 0.25

# -- Detection des sweeps (bien plus permissif que le pipeline d'origine
#    qui exigeait 11 niveaux et 500 lots) -------------------------------------
MIN_LEVELS   = 2       # sweeps de 2 ticks inclus
MIN_VOLUME   = None    # aucun plancher de volume
WINDOW_MS    = 1000
DIR_RATIO    = 0.75

# -- Fenetre de seance : indispensable pour la performance ET fidele a l'usage.
#    Heures dans le FUSEAU des timestamps du fichier (UTC en general).
#    Open cash US = 13:30 UTC en ete (EDT), 14:30 UTC en hiver (EST).
SESSION_START = "13:30"
SESSION_END   = "14:45"

# -- Confirmation du rejet (causal) -----------------------------------------
CONFIRM_SECS = 30      # duree sans depasser B pour valider un rejet

# -- Sequence ---------------------------------------------------------------
SEQ_MAX_SECS  = 600    # delai max entre la fin du sweep 1 et celle du sweep 2
GAP_MIN_TICKS = 1      # B2 doit etre au moins N ticks sous B1
GAP_MAX_TICKS = 40     # au-dela, les deux sweeps n'ont plus de rapport

# -- Trade ------------------------------------------------------------------
STOP_TICKS    = 1      # stop a STOP_TICKS au-dela de B2
HORIZON_MIN   = 60     # duree max de suivi apres l'entree

# Tout l'interet du setup est le stop serre contre B2. Si le temps de
# confirmation a laisse le prix s'eloigner, le trade n'a plus le meme profil :
# on le refuse, comme on le laisserait passer a l'ecran.
MAX_RISQUE_TICKS = 8


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def charger(path: str) -> pd.DataFrame:
    """Reutilise le loader du projet (parsing FR + cote agresseur bid/ask)."""
    from analyze_sweeps import load_ticks
    return load_ticks(path)


def filtrer_session(df: pd.DataFrame, debut: str, fin: str) -> pd.DataFrame:
    t = df.index.time
    d, f = pd.Timestamp(debut).time(), pd.Timestamp(fin).time()
    return df[(t >= d) & (t <= f)]


# ---------------------------------------------------------------------------
# Rejet causal
# ---------------------------------------------------------------------------

def confirmer_rejets(sweeps, px, tns, sens: str, confirm_ns: int):
    """
    Pour chaque sweep du sens voulu, teste si son terminus B tient.

    Rejet confirme si, pendant confirm_ns apres la fin du sweep, AUCUNE
    execution ne depasse B. La decision ne regarde que cette fenetre : elle
    serait prenable en direct.

    Retourne la liste des rejets [(sweep, B, t_confirmation_ns)].
    """
    rejets = []
    for s in sweeps:
        if s.direction != sens:
            continue
        B = s.price_high if sens == "buy" else s.price_low
        fin_ns = s.end_time.value
        t_conf = fin_ns + confirm_ns

        lo = int(np.searchsorted(tns, fin_ns, side="right"))
        hi = int(np.searchsorted(tns, t_conf, side="right"))
        if hi <= lo:
            continue                      # aucune execution : rien a confirmer

        seg = px[lo:hi]
        depasse = (seg > B).any() if sens == "buy" else (seg < B).any()
        if not depasse:
            rejets.append((s, B, t_conf))
    return rejets


def apparier(rejets, sens: str, tick: float):
    """
    Apparie deux rejets consecutifs de meme sens avec un terminus degradant :
    B2 < B1 pour des sweeps acheteurs (la demande va de moins en moins loin).

    Chaque rejet ne sert qu'une fois, en paire avec le suivant immediat.
    """
    paires = []
    seq_max_ns = SEQ_MAX_SECS * NS
    i = 0
    while i < len(rejets) - 1:
        s1, B1, _ = rejets[i]
        s2, B2, t2 = rejets[i + 1]

        ecart_ns = s2.end_time.value - s1.end_time.value
        gap = (B1 - B2) / tick if sens == "buy" else (B2 - B1) / tick

        if ecart_ns <= seq_max_ns and GAP_MIN_TICKS <= gap <= GAP_MAX_TICKS:
            paires.append((s1, B1, s2, B2, t2, gap))
            i += 2                        # les deux sweeps sont consommes
        else:
            i += 1
    return paires


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simuler(paires, px, tns, sens: str, tick: float,
            stop_ticks=STOP_TICKS, max_risque=MAX_RISQUE_TICKS,
            horizon_min=HORIZON_MIN):
    """
    Entree au moment de la confirmation du 2e rejet, stop au-dela de B2.

    Mesure MFE (excursion favorable max) et MAE (defavorable max) en ticks,
    jusqu'au stop ou a la fin de l'horizon. C'est la distribution du MFE qui
    dit ou placer les sorties — pas une regle choisie a l'avance.
    """
    short = sens == "buy"                 # rejet acheteur -> on vend
    horizon_ns = horizon_min * 60 * NS
    lignes = []
    sans_suite = trop_loin = 0

    for s1, B1, s2, B2, t_entree, gap in paires:
        lo = int(np.searchsorted(tns, t_entree, side="right"))
        hi = int(np.searchsorted(tns, t_entree + horizon_ns, side="right"))
        if hi <= lo:
            sans_suite += 1               # plus de tape apres l'entree
            continue

        entree = float(px[lo - 1]) if lo > 0 else float(px[lo])
        stop = B2 + stop_ticks * tick if short else B2 - stop_ticks * tick
        risque = abs(stop - entree) / tick
        if risque > max_risque:
            trop_loin += 1
            continue

        seg = px[lo:hi]
        segt = tns[lo:hi]

        # Le stop coupe la trajectoire : tout ce qui suit ne compte pas.
        touche = (seg >= stop) if short else (seg <= stop)
        if touche.any():
            j = int(np.argmax(touche))
            vie, viet = seg[:j + 1], segt[:j + 1]
            stoppe = True
        else:
            vie, viet = seg, segt
            stoppe = False

        if short:
            mfe = (entree - float(vie.min())) / tick
            mae = (float(vie.max()) - entree) / tick
            j_mfe = int(np.argmin(vie))
        else:
            mfe = (float(vie.max()) - entree) / tick
            mae = (entree - float(vie.min())) / tick
            j_mfe = int(np.argmax(vie))

        lignes.append({
            "t_sweep1": s1.end_time, "B1": B1,
            "t_entree": pd.Timestamp(t_entree), "B2": B2,
            "gap_ticks": round(gap, 1),
            "ecart_secs": round((s2.end_time.value - s1.end_time.value) / NS, 1),
            "niveaux_1": s1.levels_crossed, "niveaux_2": s2.levels_crossed,
            "vol_1": s1.total_volume, "vol_2": s2.total_volume,
            "entree": entree, "stop": stop,
            "risque_ticks": round(risque, 1),
            "mfe_ticks": round(max(0.0, mfe), 1),
            "mae_ticks": round(max(0.0, mae), 1),
            "t_to_mfe_s": round((int(viet[j_mfe]) - t_entree) / NS, 1),
            "stoppe": stoppe,
        })

    if trop_loin or sans_suite:
        print(f"  {trop_loin} sequence(s) ecartee(s) : entree a plus de "
              f"{max_risque} ticks du stop"
              + (f", {sans_suite} sans tape ensuite" if sans_suite else ""))
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def rapport(df: pd.DataFrame):
    n = len(df)
    print("\n" + "=" * 64)
    print(f"SEQUENCES DE DOUBLE REJET : {n}")
    print("=" * 64)
    if not n:
        print("\nAucune sequence. Desserre CONFIRM_SECS, SEQ_MAX_SECS ou GAP_MAX_TICKS.")
        return

    print(f"\nRisque median (entree -> stop) : {df['risque_ticks'].median():.1f} ticks")
    print(f"Taux de stop                   : {df['stoppe'].mean():.0%}")

    print("\n--- Distribution du MFE (excursion favorable max) ---")
    print("C'est ce tableau qui dit ou sortir.")
    bornes = [-0.1, 0, 2, 4, 8, 16, 40, 1e9]
    noms = ["0", "1-2", "3-4", "5-8", "9-16", "17-40", "40+"]
    buckets = pd.cut(df["mfe_ticks"], bins=bornes, labels=noms)
    cum = 0
    for nom in noms:
        c = int((buckets == nom).sum())
        cum += c
        print(f"  {nom:>6} ticks : {c:>5}  ({100*c/n:5.1f} %)   cumul {100*cum/n:5.1f} %")

    print("\n--- Atteinte de paliers (part des trades) ---")
    for cible in (2, 4, 8, 12, 20, 40):
        atteint = (df["mfe_ticks"] >= cible).mean()
        # Esperance d'une sortie seche a ce palier, stop = risque median
        risque = df["risque_ticks"].median()
        esp = atteint * cible - (1 - atteint) * risque
        print(f"  >= {cible:>3} ticks : {atteint:6.1%}   esperance sortie seche "
              f"{esp:+6.2f} ticks")

    print(f"\nMediane du temps jusqu'au MFE : {df['t_to_mfe_s'].median():.0f} s")
    print("  -> si c'est court, un stop temporel elimine les trades qui trainent.")

    print("\n--- Apercu ---")
    cols = ["t_entree", "B1", "B2", "gap_ticks", "ecart_secs",
            "risque_ticks", "mfe_ticks", "mae_ticks", "t_to_mfe_s", "stoppe"]
    with pd.option_context("display.max_rows", 15, "display.width", 200):
        print(df[cols].head(15).to_string(index=False))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fichier")
    ap.add_argument("--sens", choices=["buy", "sell"], default="buy",
                    help="buy = rejets acheteurs -> shorts (defaut)")
    ap.add_argument("--debut", default=SESSION_START)
    ap.add_argument("--fin", default=SESSION_END)
    ap.add_argument("--confirm", type=int, default=CONFIRM_SECS)
    ap.add_argument("--niveaux", type=int, default=MIN_LEVELS)
    ap.add_argument("--risque", type=float, default=MAX_RISQUE_TICKS,
                    help="ecart max entree-stop en ticks (defaut 8)")
    ap.add_argument("--stop", type=float, default=STOP_TICKS,
                    help="stop a N ticks au-dela de B2 (defaut 1)")
    ap.add_argument("--horizon", type=int, default=HORIZON_MIN,
                    help="duree max de suivi en minutes (defaut 60)")
    args = ap.parse_args()

    print(f"Chargement de {args.fichier} ...")
    full = charger(args.fichier)
    print(f"  {len(full):,} ticks — {full.index[0]} -> {full.index[-1]}")

    session = filtrer_session(full, args.debut, args.fin)
    part = len(session) / len(full) if len(full) else 0
    print(f"  Session {args.debut}-{args.fin} : {len(session):,} ticks ({part:.1%})")
    if part < 0.02:
        print("  /!\\ Tres peu de ticks : le fuseau ne colle probablement pas.")
        print("      Verifie l'heure du pic de volume et ajuste --debut/--fin.")
        sys.exit(1)

    print(f"\nDetection des sweeps (>={args.niveaux} niveaux, volume libre) ...")
    sweeps = detect_tape_sweeps(
        session, window_ms=WINDOW_MS, min_levels=args.niveaux,
        min_volume=MIN_VOLUME, tick_size=TICK, min_directional_ratio=DIR_RATIO,
    )
    n_sens = sum(1 for s in sweeps if s.direction == args.sens)
    print(f"  {len(sweeps):,} sweeps dont {n_sens:,} en '{args.sens}'")
    if not sweeps:
        return

    # Le suivi doit se faire sur la tape COMPLETE : un trade ouvert en fin de
    # fenetre continue de vivre apres la borne de session.
    px  = np.asarray(full["price"].to_numpy(), dtype=np.float64)
    tns = full.index.values.astype("datetime64[ns]").astype("int64")

    rejets = confirmer_rejets(sweeps, px, tns, args.sens, args.confirm * NS)
    print(f"  {len(rejets):,} rejets confirmes ({args.confirm}s sans depassement)")

    paires = apparier(rejets, args.sens, TICK)
    print(f"  {len(paires):,} sequences de double rejet")

    df = simuler(paires, px, tns, args.sens, TICK,
                 stop_ticks=args.stop, max_risque=args.risque,
                 horizon_min=args.horizon)
    rapport(df)

    if len(df):
        out = Path(args.fichier).with_name(f"rejets_{args.sens}.csv")
        df.to_csv(out, index=False)
        print(f"\nExporte -> {out}")


if __name__ == "__main__":
    main()

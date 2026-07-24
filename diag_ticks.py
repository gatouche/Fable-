#!/usr/bin/env python3
"""
Diagnostic de qualite des donnees tick ES.

Objectif : determiner si le fichier permet de detecter les sweeps
above-ask / below-bid, AVANT de construire le detecteur de pattern.

Lecture par blocs : memoire constante, fonctionne sur des fichiers de
plusieurs Go.

Usage :
    python diag_ticks.py ticks_ES_full.csv
    python diag_ticks.py ticks_ES_full.csv 500000     # taille de bloc

Dependances : pandas
    pip install pandas
"""

import sys
from collections import Counter

import pandas as pd

TICK = 0.25                      # taille du tick ES
BLOC = 1_000_000                 # lignes par bloc
FMT = "%Y-%m-%d %H:%M:%S.%f"     # format horodatage attendu
COLS = ["time", "last", "bid", "ask", "volume"]
LABELS = ["below_bid", "at_bid", "between", "at_ask", "above_ask"]


def preparer(brut):
    """Parse un bloc et classe chaque execution dans les 5 buckets du tape."""
    df = brut.copy()

    # format= explicite : sans lui, le parsing de millions de dates est
    # environ 20x plus lent
    try:
        df["time"] = pd.to_datetime(df["time"], format=FMT)
    except ValueError:
        df["time"] = pd.to_datetime(df["time"])

    # Travailler en ticks entiers evite tous les pieges de comparaison flottante
    for col in ("last", "bid", "ask"):
        df[col] = (df[col] / TICK).round().astype("int64")

    last, bid, ask = df["last"], df["bid"], df["ask"]
    conditions = [
        last < bid,
        last == bid,
        (last > bid) & (last < ask),
        last == ask,
        last > ask,
    ]

    df["bucket"] = pd.NA
    for cond, label in zip(conditions, LABELS):
        df.loc[cond & df["bucket"].isna(), "bucket"] = label

    return df


def agreger(df):
    """
    Regroupe les lignes partageant le meme timestamp en un evenement unique.

    Le flux eclate chaque transaction en lignes d'un lot ; sans ce regroupement
    toute analyse de taille est fausse et les sweeps multi-niveaux sont
    invisibles.
    """
    grp = df.groupby("time", sort=False)
    return pd.DataFrame({
        "volume": grp["volume"].sum(),
        "n_lignes": grp.size(),
        "niveaux": grp["last"].max() - grp["last"].min() + 1,
    })


class Accumulateur:
    """Statistiques cumulees bloc par bloc, a memoire bornee."""

    def __init__(self):
        self.n = 0
        self.buckets = Counter()
        self.heures = Counter()
        self.dates = set()
        self.t_min = self.t_max = None
        self.tout_volume_1 = True
        self.precision_ns = 10 ** 9      # plus petite granularite constatee
        self.n_evenements = 0
        self.vol_max = 0
        self.lignes_max = 0
        self.sweeps = Counter()          # largeur en ticks -> nombre

    def ajouter_lignes(self, df):
        self.n += len(df)
        self.buckets.update(df["bucket"].dropna())
        self.heures.update(df["time"].dt.hour)
        self.dates.update(df["time"].dt.date.unique())

        t0, t1 = df["time"].min(), df["time"].max()
        self.t_min = t0 if self.t_min is None else min(self.t_min, t0)
        self.t_max = t1 if self.t_max is None else max(self.t_max, t1)

        if self.tout_volume_1 and not (df["volume"] == 1).all():
            self.tout_volume_1 = False

        # Normalisation explicite en ns : pandas 2.x peut stocker en us,
        # auquel cas astype("int64") renverrait des microsecondes.
        ns = df["time"].astype("datetime64[ns]").astype("int64")
        for granularite in (10 ** 9, 10 ** 6, 10 ** 3):
            if granularite <= self.precision_ns and not (ns % granularite == 0).all():
                self.precision_ns = granularite // 1000

    def ajouter_evenements(self, ev):
        if ev.empty:
            return
        self.n_evenements += len(ev)
        self.vol_max = max(self.vol_max, int(ev["volume"].max()))
        self.lignes_max = max(self.lignes_max, int(ev["n_lignes"].max()))
        self.sweeps.update(ev.loc[ev["niveaux"] >= 2, "niveaux"])


def parcourir(chemin, bloc):
    """Lit le fichier par blocs en gerant les evenements a cheval."""
    acc = Accumulateur()
    report = None  # lignes du dernier timestamp, potentiellement incomplet

    lecteur = pd.read_csv(chemin, chunksize=bloc, usecols=COLS)
    for i, brut in enumerate(lecteur, 1):
        df = preparer(brut)
        acc.ajouter_lignes(df)

        if report is not None:
            df = pd.concat([report, df], ignore_index=True)

        # Le dernier timestamp du bloc peut continuer dans le bloc suivant :
        # on le met de cote au lieu de l'agreger a tort.
        dernier = df["time"].iloc[-1]
        a_reporter = df["time"] == dernier
        report = df.loc[a_reporter]
        acc.ajouter_evenements(agreger(df.loc[~a_reporter]))

        print(f"  bloc {i} — {acc.n:,} lignes lues", end="\r", flush=True)

    if report is not None and not report.empty:
        acc.ajouter_evenements(agreger(report))

    print(" " * 60, end="\r")
    return acc


def rapport(acc):
    n = acc.n
    print("=" * 62)
    print("DIAGNOSTIC DONNEES TICK")
    print("=" * 62)

    print(f"\nLignes brutes        : {n:,}")
    print(f"Evenements agreges   : {acc.n_evenements:,}")
    print(f"Periode              : {acc.t_min}  ->  {acc.t_max}")
    print(f"Jours distincts      : {len(acc.dates)}")

    noms = {10 ** 9: "SECONDE (tres insuffisant)", 10 ** 6: "MILLISECONDE",
            10 ** 3: "MICROSECONDE", 1: "NANOSECONDE"}
    print(f"Precision horodatage : {noms.get(acc.precision_ns, '?')}")

    print("\n--- Activite par heure (top 6) ---")
    print("Sert a identifier le fuseau : le pic doit tomber a l'ouverture cash.")
    for h, c in sorted(acc.heures.most_common(6)):
        print(f"  {h:02d}h : {c:>12,}  ({100*c/n:5.2f} %)")

    print("\n--- Repartition des buckets (TEST DECISIF) ---")
    for label in LABELS:
        c = acc.buckets.get(label, 0)
        print(f"  {label:<10} : {c:>12,}  ({100*c/n:6.3f} %)")

    hors = acc.buckets.get("above_ask", 0) + acc.buckets.get("below_bid", 0)
    pct = 100 * hors / n

    print("\n--- VERDICT ---")
    if hors == 0:
        print("  BLOQUANT : aucun trade hors du spread.")
        print("  Le quote est presque certainement enregistre APRES le trade.")
        print("  Les sweeps sont invisibles dans ce fichier -> il faut une")
        print("  autre source (Databento GLBX.MDP3, schema mbp-1).")
    elif pct < 0.05:
        print(f"  DOUTEUX : seulement {pct:.3f} % hors spread.")
        print("  Trop rare pour un ES actif. Alignement quote/trade suspect.")
    elif pct > 15:
        print(f"  DOUTEUX : {pct:.2f} % hors spread, beaucoup trop.")
        print("  Le quote retarde sur le trade -> faux positifs massifs.")
    else:
        print(f"  EXPLOITABLE : {pct:.2f} % hors spread, ordre de grandeur sain.")
        print("  On peut construire le detecteur de rejections.")

    print("\n--- Volume ---")
    if acc.tout_volume_1:
        print("  Toutes les lignes brutes sont a 1 lot -> flux eclate confirme.")
    print(f"  Volume max par evenement : {acc.vol_max:,}")
    print(f"  Lignes max par evenement : {acc.lignes_max:,}")

    n_sweeps = sum(acc.sweeps.values())
    print("\n--- Sweeps multi-niveaux (meme timestamp, >= 2 prix) ---")
    part = 100 * n_sweeps / acc.n_evenements if acc.n_evenements else 0
    print(f"  Nombre  : {n_sweeps:,}  ({part:.3f} % des evenements)")
    if n_sweeps:
        print(f"  Largeur max : {max(acc.sweeps)} ticks")
        print("  Distribution :")
        for largeur in sorted(acc.sweeps)[:8]:
            print(f"    {largeur:>3} ticks : {acc.sweeps[largeur]:>10,}")
        print("  -> c'est ce que Jigsaw regroupe en une ligne sur ton tape.")
    else:
        print("  Aucun. Soit l'horodatage est trop grossier, soit le flux")
        print("  ne conserve pas les executions individuelles d'un sweep.")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit("Usage : python diag_ticks.py <fichier.csv> [taille_bloc]")

    bloc = int(sys.argv[2]) if len(sys.argv) == 3 else BLOC
    rapport(parcourir(sys.argv[1], bloc))

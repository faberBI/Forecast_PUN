"""
damodaran_update.py
===================
Scarica i dati ufficiali di Aswath Damodaran (Stern NYU) e li normalizza in un dataset
versionato (JSON) che damodaran_data.py caricherà automaticamente.

  Esegui:   python damodaran_update.py
  Produce:  damodaran_dataset.json  (beta di settore per regione + ERP per paese, con data)

Dipendenze: pandas, xlrd  (i file Damodaran sono .xls "vecchio formato": serve xlrd>=2.0).
  pip install pandas xlrd

Nota: il download richiede accesso a internet verso pages.stern.nyu.edu. Se l'ambiente
è isolato, esegui questo script dove la rete è disponibile e copia il JSON accanto all'app.

Il parsing è ROBUSTO: individua la riga di intestazione cercando i nomi delle colonne
(non usa posizioni fisse), così regge piccole modifiche di layout tra un'annata e l'altra.
"""

import io
import json
import sys
from datetime import date
from urllib.request import urlopen, Request

import pandas as pd

BASE = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/"
BETA_FILES = {          # regione -> file .xls
    "US": "betas.xls",
    "Europe": "betaEurope.xls",
    "Emerging": "betaemerg.xls",
    "Global": "betaglobal.xls",
    "Japan": "betaJapan.xls",
}
ERP_FILE = "ctryprem.xls"


# ==========================================================================
# utility di conversione (robuste a stringhe "6.70%" e frazioni 0.067)
# ==========================================================================
def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            f = float(x)
            return None if pd.isna(f) else f
        except Exception:
            return None
    s = str(x).strip().replace("%", "").replace(",", "")
    if s == "" or s.upper() in ("NA", "N/A", "NR"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_pct(x):
    """Restituisce PUNTI PERCENTUALI. Interpreta 0.067 come 6.70 e '6.70%' come 6.70."""
    was_pct_string = isinstance(x, str) and "%" in x
    f = _to_float(x)
    if f is None:
        return None
    if was_pct_string:
        return f                      # "6.70%" -> 6.70
    return f * 100.0 if abs(f) <= 1.0 else f   # 0.067 -> 6.70 ; 6.70 -> 6.70


def _norm(s):
    return " ".join(str(s).strip().lower().split())


def _find_header_row(raw, keyword):
    """Indice della riga di intestazione. Cerca prima una cella che EGUAGLIA la keyword
    (le celle di intestazione sono corte), poi come fallback una cella che inizia con la
    keyword ed è breve — così non scambia un titolo lungo (che 'contiene' la parola)
    per la riga di intestazione."""
    kw = _norm(keyword)
    for i in range(len(raw)):                     # 1) match esatto di cella
        for cell in raw.iloc[i].tolist():
            if _norm(cell) == kw:
                return i
    for i in range(len(raw)):                     # 2) fallback: inizia con keyword ed è breve
        for cell in raw.iloc[i].tolist():
            c = _norm(cell)
            if c.startswith(kw) and len(c) <= len(kw) + 15:
                return i
    return None


def _col_matching(header_cells, must_have, must_not=()):
    """Indice della colonna la cui intestazione contiene tutti i termini `must_have`
    e nessuno dei termini `must_not`."""
    for j, cell in enumerate(header_cells):
        h = _norm(cell)
        if all(t in h for t in must_have) and not any(t in h for t in must_not):
            return j
    return None


# ==========================================================================
# parser (testabili su griglie sintetiche: prendono un DataFrame header=None)
# ==========================================================================
def parse_beta_grid(raw):
    """Da una griglia grezza (header=None) del file beta -> {settore: (unlevered, unlevered_cash)}."""
    hr = _find_header_row(raw, "Industry Name")
    if hr is None:
        hr = _find_header_row(raw, "Industry")
    if hr is None:
        raise ValueError("Riga di intestazione non trovata (manca 'Industry Name').")
    header = raw.iloc[hr].tolist()
    c_name = _col_matching(header, ["industry"])
    c_ub = _col_matching(header, ["unlevered", "beta"], must_not=["cash"])
    c_ubc = _col_matching(header, ["unlevered", "beta", "cash"])
    if c_name is None or c_ub is None:
        raise ValueError("Colonne 'Industry'/'Unlevered beta' non individuate.")
    out = {}
    for i in range(hr + 1, len(raw)):
        name = raw.iat[i, c_name]
        if name is None or str(name).strip() == "" or _norm(name).startswith("total market"):
            continue
        ub = _to_float(raw.iat[i, c_ub])
        ubc = _to_float(raw.iat[i, c_ubc]) if c_ubc is not None else ub
        if ub is None:
            continue
        out[str(name).strip()] = (ub, ubc if ubc is not None else ub)
    return out


def parse_erp_grid(raw):
    """Da una griglia grezza (header=None) del file ctryprem -> {paese: erp_totale_pct}."""
    hr = _find_header_row(raw, "Country")
    if hr is None:
        raise ValueError("Riga di intestazione non trovata (manca 'Country').")
    header = raw.iloc[hr].tolist()
    c_country = _col_matching(header, ["country"])
    # preferisci 'Total Equity Risk Premium'; in mancanza, 'Equity Risk Premium'
    c_erp = _col_matching(header, ["total", "equity", "risk", "premium"])
    if c_erp is None:
        c_erp = _col_matching(header, ["equity", "risk", "premium"])
    if c_country is None or c_erp is None:
        raise ValueError("Colonne 'Country'/'Equity Risk Premium' non individuate.")
    out = {}
    for i in range(hr + 1, len(raw)):
        name = raw.iat[i, c_country]
        if name is None or str(name).strip() == "":
            continue
        val = _to_pct(raw.iat[i, c_erp])
        if val is None:
            continue
        out[str(name).strip()] = round(val, 3)
    return out


# ==========================================================================
# download
# ==========================================================================
def _fetch_grid(url, timeout=60):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (DCF tool updater)"})
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    # .xls vecchio formato -> engine xlrd; header=None per leggere la griglia grezza
    return pd.read_excel(io.BytesIO(data), header=None, engine="xlrd")


def build_dataset(out_path="damodaran_dataset.json", verbose=True):
    betas = {}
    for region, fname in BETA_FILES.items():
        url = BASE + fname
        try:
            raw = _fetch_grid(url)
            betas[region] = parse_beta_grid(raw)
            if verbose:
                print(f"  {region:9}: {len(betas[region])} settori  <- {fname}")
        except Exception as e:
            if verbose:
                print(f"  {region:9}: ERRORE ({e})")
    try:
        erp = parse_erp_grid(_fetch_grid(BASE + ERP_FILE))
        if verbose:
            print(f"  ERP      : {len(erp)} paesi  <- {ERP_FILE}")
    except Exception as e:
        erp = {}
        if verbose:
            print(f"  ERP      : ERRORE ({e})")

    dataset = {
        "metadata": {
            "source": "Aswath Damodaran, Stern NYU (pages.stern.nyu.edu/~adamodar)",
            "date": date.today().isoformat(),
            "base_url": BASE,
        },
        "betas": {r: {s: [ub, ubc] for s, (ub, ubc) in secs.items()} for r, secs in betas.items()},
        "erp": erp,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"Scritto {out_path} (data {dataset['metadata']['date']}).")
    return dataset


if __name__ == "__main__":
    print("Aggiornamento dati Damodaran...")
    try:
        build_dataset()
    except Exception as e:
        print(f"Aggiornamento fallito: {e}", file=sys.stderr)
        print("Verifica connessione a pages.stern.nyu.edu e che 'xlrd' sia installato "
              "(pip install xlrd).", file=sys.stderr)
        sys.exit(1)

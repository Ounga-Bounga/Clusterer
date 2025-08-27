import re
import io
import unicodedata
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SEO Clustering App", page_icon="🔗", layout="wide")
st.title("🔗 SEO Clustering (SERP Similarity)")

st.write(
    "Dépose un CSV **ou** un XLSX. L’app : "
    "1) calcule les métriques globales, "
    "2) retire les accents et **dé-duplique** (on garde le plus gros volume), "
    "3) clusterise selon le seuil de similarité (en %)."
)

# ---------- Helpers ----------
def strip_accents(text: str) -> str:
    """Supprime les accents (sans toucher aux autres caractères)."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text

def normalize_kw(text: str) -> str:
    """Normalise pour comparer: accents retirés, lower, espaces compactés/trim."""
    base = strip_accents(text).lower().strip()
    base = re.sub(r"\s+", " ", base)
    return base

@st.cache_data(show_spinner=False)
def parse_similarity_cell(cell: str):
    """
    Transforme une cellule 'KW list and %' en liste [(keyword, percent_float), ...]
    Gère p.ex.:
      'kw (1600): 40%'  'kw: 40 %'  'kw 40%'  séparés par '|'
    """
    if pd.isna(cell) or not str(cell).strip():
        return []
    parts = [p.strip() for p in str(cell).split("|")]
    results = []
    for p in parts:
        p = re.sub(r"\s*%\s*$", "%", p.strip())  # normaliser espaces avant %
        # Forme "kw (123): 40%"
        m = re.match(r"(.+?)(?:\(\s*\d+\s*\))?\s*:\s*([\d.,]+)\s*%", p)
        if m:
            kw = m.group(1).strip()
            pct = m.group(2).replace(",", ".").strip()
            try:
                results.append((kw, float(pct)))
            except:
                pass
            continue
        # Forme fallback "kw 40%"
        m2 = re.match(r"(.+?)\s+([\d.,]+)\s*%", p)
        if m2:
            kw = m2.group(1).strip()
            pct = m2.group(2).replace(",", ".").strip()
            try:
                results.append((kw, float(pct)))
            except:
                pass
    return results

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

def _read_any(file):
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def prepare_base(df: pd.DataFrame):
    """
    - Normalise colonnes
    - Calcule métriques globales (count / volume total)
    - Retire accents + dé-duplique (garde la ligne au volume max par forme normalisée)
    - Construit un alias_map: norm -> mot-clé canonique conservé
    - Renvoie aussi le nombre de mots-clés retirés par déduplication
    """
    cols = {c.lower(): c for c in df.columns}
    kw_col = cols.get("keyword") or list(df.columns)[0]
    vol_col = cols.get("monthly vol.") or cols.get("volume") or list(df.columns)[1]
    sim_col = cols.get("kw list and %") or list(df.columns)[2]

    data = df.copy()
    data[kw_col] = data[kw_col].astype(str).str.strip()
    data[vol_col] = pd.to_numeric(data[vol_col], errors="coerce").fillna(0).astype(float)

    # métriques globales "raw" (avant dédup)
    file_kw_count_raw = int(data[kw_col].nunique())
    file_total_volume = int(data[vol_col].sum())

    # clé de dédup (accents out + lower + espaces compactés)
    data["__norm"] = data[kw_col].map(normalize_kw)

    # garder index de la ligne au volume max dans chaque groupe normalisé
    idx_keep = data.groupby("__norm")[vol_col].idxmax()
    deduped = data.loc[idx_keep].copy()

    # alias map: norm -> keyword canonique (texte de la ligne conservée)
    norm_to_canonical = dict(zip(deduped["__norm"], deduped[kw_col]))

    # compter les doublons supprimés
    file_kw_count_dedup = int(deduped["__norm"].nunique())
    removed_dups = file_kw_count_raw - file_kw_count_dedup

    # on peut drop la colonne technique
    deduped = deduped.drop(columns=["__norm"])

    return (
        deduped,
        kw_col,
        vol_col,
        sim_col,
        file_kw_count_raw,
        file_total_volume,
        removed_dups,
        norm_to_canonical,
    )

@st.cache_data(show_spinner=True)
def clusterize(data: pd.DataFrame, kw_col: str, vol_col: str, sim_col: str,
               threshold: float, norm_to_canonical: dict):
    """
    Résultat AU FORMAT DEMANDÉ :
    - main_keyword
    - main_volume
    - keywords_count
    - cluster_volume
    - secondary_keywords (sans le main, séparés par " | ")
    """
    # volumes/corpus après déduplication
    volumes = dict(zip(data[kw_col], data[vol_col]))
    all_kws = list(data[kw_col])

    uf = UnionFind()
    for kw in all_kws:
        uf.add(kw)

    # Créer un resolver d'alias: texte -> canonique
    def resolve_canonical(name: str) -> str | None:
        norm = normalize_kw(name)
        canon = norm_to_canonical.get(norm)
        return canon if canon in volumes else None

    # Construire les arêtes (un sens suffit)
    for _, row in data.iterrows():
        a = row[kw_col]
        sims = parse_similarity_cell(row.get(sim_col, ""))
        for b_raw, pct in sims:
            if pct < threshold:
                continue
            b = resolve_canonical(b_raw)
            if b is None:
                continue
            uf.add(b)
            uf.union(a, b)

    # Groupes
    groups = {}
    for kw in all_kws:
        root = uf.find(kw)
        groups.setdefault(root, []).append(kw)

    # Résumé au format demandé
    out_rows = []
    for _, members in groups.items():
        members_sorted = sorted(members, key=lambda k: volumes.get(k, 0), reverse=True)
        main_kw = members_sorted[0]
        main_vol = int(volumes.get(main_kw, 0))
        cluster_vol = int(sum(volumes.get(k, 0) for k in members_sorted))
        secondary = [k for k in members_sorted if k != main_kw]
        out_rows.append({
            "main_keyword": main_kw,
            "main_volume": main_vol,
            "keywords_count": len(members_sorted),
            "cluster_volume": cluster_vol,
            "secondary_keywords": " | ".join(secondary),
        })

    result = pd.DataFrame(out_rows).sort_values(
        ["keywords_count", "cluster_volume"], ascending=[False, False]
    ).reset_index(drop=True)

    return result

# ---------- UI ----------
left, right = st.columns([1, 2])

with left:
    st.subheader("1) Dépose ton fichier")
    file = st.file_uploader(
        "CSV ou XLSX avec: Keyword | Monthly vol. | KW list and %",
        type=["csv", "xlsx"]
    )
    threshold = st.slider("Seuil de similarité (%)", min_value=0, max_value=100, value=30, step=5)
    st.caption("Astuce: 30–40% marche bien pour des SERP FR.")

if file:
    df = _read_any(file)
    (
        data, kw_col, vol_col, sim_col,
        file_kw_count_raw, file_total_volume,
        removed_dups, norm_to_canonical
    ) = prepare_base(df)

    with st.expander("Aperçu (10 premières lignes après dédup)"):
        st.dataframe(data.head(10), use_container_width=True)

    # Métriques globales
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Mots-clés (fichier RAW)", value=f"{file_kw_count_raw}")
    with m2:
        st.metric("Volume total (fichier RAW)", value=f"{file_total_volume:,}".replace(",", " "))
    with m3:
        st.metric("Doublons supprimés (sans accents)", value=f"{removed_dups}")

    # Clusterisation dynamique (sur le corpus dédupliqué)
    result_df = clusterize(
        data, kw_col, vol_col, sim_col, float(threshold), norm_to_canonical
    )

    with right:
        st.subheader("2) Résultats des clusters")
        st.dataframe(result_df, use_container_width=True, height=560)

        # Exports
        # Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Clusters")
        st.download_button(
            label="⬇️ Télécharger (Excel .xlsx)",
            data=output.getvalue(),
            file_name="seo_clusters_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        # CSV
        st.download_button(
            label="⬇️ Télécharger (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="seo_clusters_summary.csv",
            mime="text/csv",
        )
else:
    st.info("Dépose un CSV/XLSX pour lancer le clustering.")

import re
import io
import unicodedata
import pandas as pd
import streamlit as st

# =============== App config ===============
st.set_page_config(page_title="SEO Clustering → Ranking Decisions", page_icon="🔗", layout="wide")
st.title("🔗 SEO Clustering (SERP Similarity) → 📈 Ranking decisions")

st.write(
    "1) Dépose ton **fichier mots-clés** (CSV/XLSX) → dédup **sans accents** → métriques → clusters avec **seuil de similarité**.\n"
    "2) Dépose ton **fichier de rankings** (CSV/XLSX) → décision **Optimize (1–20)** ou **Create new page**."
)

# =========================================
#                 Utils
# =========================================
def strip_accents(text: str) -> str:
    """Supprime les accents, ne modifie pas le reste."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def normalize_kw(text: str) -> str:
    """Forme canonique pour comparaison: accents out + lower + espaces compactés/trim."""
    base = strip_accents(text).lower().strip()
    base = re.sub(r"\s+", " ", base)
    return base

def _read_any(file):
    """Lit CSV/XLSX. Pour XLSX, lit le 1er onglet par défaut (comme la plupart des exports)."""
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file, sheet_name=0)
    return pd.read_csv(file)

# =========================================
#       Parsing de la colonne similarités
# =========================================
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

# =========================================
#                Union-Find
# =========================================
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

# =========================================
#     Step 1 : Base (dédup sans accents)
# =========================================
@st.cache_data(show_spinner=True)
def prepare_base(df: pd.DataFrame):
    """
    - Normalise colonnes
    - Calcule métriques RAW (count/volume)
    - Dédup: supprime accents + minuscule pour comparer; conserve la ligne au volume max
    - Retourne aussi: removed_dups et norm_to_canonical (norm -> kw retenu)
    """
    cols = {c.lower(): c for c in df.columns}
    kw_col = cols.get("keyword") or list(df.columns)[0]
    vol_col = cols.get("monthly vol.") or cols.get("volume") or list(df.columns)[1]
    sim_col = cols.get("kw list and %") or list(df.columns)[2]

    data = df.copy()
    data[kw_col] = data[kw_col].astype(str).str.strip()
    data[vol_col] = pd.to_numeric(data[vol_col], errors="coerce").fillna(0).astype(float)

    # Métriques RAW
    file_kw_count_raw = int(data[kw_col].nunique())
    file_total_volume = int(data[vol_col].sum())

    # Dédup par forme normalisée
    data["__norm"] = data[kw_col].map(normalize_kw)
    idx_keep = data.groupby("__norm")[vol_col].idxmax()
    deduped = data.loc[idx_keep].copy()
    norm_to_canonical = dict(zip(deduped["__norm"], deduped[kw_col]))

    removed_dups = file_kw_count_raw - int(deduped["__norm"].nunique())
    deduped = deduped.drop(columns=["__norm"])

    return (
        deduped, kw_col, vol_col, sim_col,
        file_kw_count_raw, file_total_volume,
        removed_dups, norm_to_canonical
    )

# =========================================
#         Step 1b : Clustering
# =========================================
@st.cache_data(show_spinner=True)
def clusterize(data: pd.DataFrame, kw_col: str, vol_col: str, sim_col: str,
               threshold: float, norm_to_canonical: dict):
    """
    Sortie: tableau UNIQUE au format demandé :
      - main_keyword (nom du cluster)
      - main_volume
      - keywords_count (main + secondaires)
      - cluster_volume (somme volumes du cluster)
      - secondary_keywords (sans le main, séparés par " | ")
    """
    volumes = dict(zip(data[kw_col], data[vol_col]))
    all_kws = list(data[kw_col])
    uf = UnionFind()
    for kw in all_kws:
        uf.add(kw)

    def resolve_canonical(name: str):
        norm = normalize_kw(name)
        canon = norm_to_canonical.get(norm)
        return canon if canon in volumes else None

    # Arêtes (un sens suffit)
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

    # Résumé
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

# =========================================
#          Step 2 : Rankings
# =========================================
@st.cache_data(show_spinner=True)
def prepare_ranking(df_rank: pd.DataFrame):
    """
    Normalise le fichier de rankings.
    Colonnes attendues (noms souples) :
      - keyword  (obligatoire)
      - url      (obligatoire)
      - position (obligatoire) : accepte 'position', 'rank', 'pos', 'google position', etc.
    """
    cols = {c.lower().strip(): c for c in df_rank.columns}

    # keyword
    kw_candidates = ["keyword", "kw", "query", "search term"]
    kwc = next((cols[c] for c in kw_candidates if c in cols), None)
    if not kwc:
        kwc = list(df_rank.columns)[0]

    # url
    url_candidates = ["url", "landing page", "page", "target url"]
    urlc = next((cols[c] for c in url_candidates if c in cols), None)
    if not urlc:
        urlc = list(df_rank.columns)[1] if len(df_rank.columns) > 1 else kwc

    # position
    pos_candidates = ["position", "rank", "pos", "google position", "google rank"]
    posc = next((cols[c] for c in pos_candidates if c in cols), None)
    if not posc:
        posc = list(df_rank.columns)[2] if len(df_rank.columns) > 2 else None
        if posc is None:
            raise ValueError("Impossible de trouver la colonne de position (position/rank/pos).")

    r = df_rank.copy()
    r[kwc] = r[kwc].astype(str).str.strip()
    r[urlc] = r[urlc].astype(str).str.strip()
    r[posc] = pd.to_numeric(r[posc], errors="coerce")

    r = r.dropna(subset=[kwc, urlc, posc])
    r["__norm_kw"] = r[kwc].map(normalize_kw)

    return r[[kwc, urlc, posc, "__norm_kw"]].rename(
        columns={kwc: "rank_keyword", urlc: "rank_url", posc: "rank_position"}
    )

@st.cache_data(show_spinner=True)
def decide_from_ranking(clusters_df: pd.DataFrame, ranking_norm_df: pd.DataFrame, optimize_max_pos: int = 20):
    """
    Mappe chaque main_keyword -> ranking rows (via normalisation sans accents).
    Décision:
      - best position in [1, optimize_max_pos]  => Optimize existing page
      - else (>optimize_max_pos or no data)     => Create new page
    Sortie: clusters_df + colonnes:
      - best_position
      - best_url
      - decision
      - evidence (top 3 "pos — url")
    """
    # index par kw normalisé
    groups = {}
    for _, row in ranking_norm_df.iterrows():
        groups.setdefault(row["__norm_kw"], []).append((row["rank_position"], row["rank_url"]))

    rows = []
    for _, r in clusters_df.iterrows():
        mk = r["main_keyword"]
        norm = normalize_kw(mk)
        matches = groups.get(norm, [])

        best_pos, best_url = None, None
        evidence_list = []
        if matches:
            # tri par position croissante (1 = meilleur)
            matches_sorted = sorted([m for m in matches if pd.notna(m[0])])
            if matches_sorted:
                best_pos, best_url = matches_sorted[0]
                evidence_list = [f"{int(p)} — {u}" for p, u in matches_sorted[:3] if pd.notna(p)]

        if best_pos is not None and 1 <= best_pos <= optimize_max_pos:
            decision = "Optimize existing page"
        elif best_pos is not None and best_pos > optimize_max_pos:
            decision = "Create new page"
        else:
            decision = "Create new page (no ranking)"

        rows.append({
            **r.to_dict(),
            "best_position": int(best_pos) if best_pos is not None else None,
            "best_url": best_url,
            "decision": decision,
            "evidence": " | ".join(evidence_list),
        })

    out = pd.DataFrame(rows)

    # compteurs utiles
    n_opt = int((out["decision"] == "Optimize existing page").sum())
    n_create = int((out["decision"].str.startswith("Create new page")).sum())

    return out, n_opt, n_create

# =========================================
#                   UI
# =========================================
with st.sidebar:
    st.markdown("### ℹ️ Tips")
    st.markdown("- **Dédup**: accents retirés + minuscule, conserve le **plus gros volume**.")
    st.markdown("- **Similarité**: un sens suffit (A→B ≥ seuil).")
    st.markdown("- **Decision ranking**: **1–20 = Optimize**, sinon **Create new page**.")
    optimize_max_pos = st.number_input("Seuil Optimize (meilleure position ≤)", min_value=1, max_value=100, value=20, step=1)

# --- Section 1: corpus + clusters ---
left, right = st.columns([1, 2])

with left:
    st.subheader("1) Dépose ton **fichier mots-clés**")
    file = st.file_uploader(
        "CSV/XLSX avec: Keyword | Monthly vol. | KW list and %",
        type=["csv", "xlsx"], key="kwfile"
    )
    threshold = st.slider("Seuil de similarité (%)", 0, 100, 30, 5)
    st.caption("Astuce: 30–40% marche bien pour des SERP FR.")

if file:
    # Lecture & préparation
    df = _read_any(file)
    (
        data, kw_col, vol_col, sim_col,
        file_kw_count_raw, file_total_volume,
        removed_dups, norm_to_canonical
    ) = prepare_base(df)

    with st.expander("Aperçu (10 premières lignes **après dédup**)"):
        st.dataframe(data.head(10), use_container_width=True)

    # Métriques globales (RAW)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Mots-clés (fichier RAW)", value=f"{file_kw_count_raw}")
    with m2:
        st.metric("Volume total (fichier RAW)", value=f"{file_total_volume:,}".replace(",", " "))
    with m3:
        st.metric("Doublons supprimés (sans accents)", value=f"{removed_dups}")

    # Clustering dynamique
    clusters_df = clusterize(data, kw_col, vol_col, sim_col, float(threshold), norm_to_canonical)

    with right:
        st.subheader("→ Résultats des clusters")
        st.dataframe(clusters_df, use_container_width=True, height=420)

        # Exports clusters
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            clusters_df.to_excel(w, index=False, sheet_name="Clusters")
        st.download_button(
            "⬇️ Export clusters (XLSX)", buf.getvalue(),
            file_name="seo_clusters_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "⬇️ Export clusters (CSV)",
            clusters_df.to_csv(index=False).encode("utf-8"),
            file_name="seo_clusters_summary.csv",
            mime="text/csv",
        )

    st.markdown("---")
    # --- Section 2: Ranking & decisions ---
    st.subheader("2) Dépose ton **fichier de rankings** (Semrush/Ahrefs/Monitorank, etc.)")
    st.caption("Colonnes attendues (souples) : **Keyword**, **URL**, **Position/Rank**. CSV/XLSX accepté.")
    rank_file = st.file_uploader("Ranking file", type=["csv", "xlsx"], key="rankfile")

    if rank_file:
        rank_df_raw = _read_any(rank_file)
        rank_df = prepare_ranking(rank_df_raw)

        with st.expander("Aperçu ranking (top 15)"):
            st.dataframe(rank_df.head(15), use_container_width=True)

        decisions_df, n_opt, n_create = decide_from_ranking(clusters_df, rank_df, optimize_max_pos=optimize_max_pos)

        # Appariement : combien de main keywords ont trouvé un ranking ?
        matched = decisions_df["best_position"].notna().sum()
        unmatched = len(decisions_df) - matched

        c0, c1, c2 = st.columns(3)
        with c0:
            st.metric("🔎 Main KW appariés (ranking trouvé)", matched)
        with c1:
            st.metric("✅ Optimize existing page (≤ {})".format(optimize_max_pos), n_opt)
        with c2:
            st.metric("🆕 Create new page (no/ >{})".format(optimize_max_pos), n_create)

        if unmatched > 0:
            with st.expander("Voir les main keywords **sans** ranking"):
                st.dataframe(
                    decisions_df.loc[decisions_df["best_position"].isna(), ["main_keyword"]],
                    use_container_width=True
                )

        st.subheader("→ Tableau décisionnel")
        st.dataframe(decisions_df, use_container_width=True, height=560)

        # Exports decisions
        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as w:
            decisions_df.to_excel(w, index=False, sheet_name="Decisions")
        st.download_button(
            "⬇️ Export decisions (XLSX)", buf2.getvalue(),
            file_name="seo_cluster_ranking_decisions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "⬇️ Export decisions (CSV)",
            decisions_df.to_csv(index=False).encode("utf-8"),
            file_name="seo_cluster_ranking_decisions.csv",
            mime="text/csv",
        )
else:
    st.info("Dépose d’abord le fichier mots-clés pour générer les clusters, puis ajoute ton fichier de rankings.")
with st.expander("📌 Rappel du format attendu"):
    st.markdown("""
### ✅ Résumé des formats

**File 1 (clusters)**  
Colonnes requises :  
- `Keyword`  
- `Monthly vol.`  
- `KW list and %`  

Exemple :  
```csv
Keyword,Monthly vol.,KW list and %
néons,5400,deco neon (1600): 20% | neon deco (1600): 20% | néon déco (1600): 20%
tube à led,4400,neons led (1600): 40% | neon led (1600): 30% | led neon (1600): 10%
deco neon,1600,neon deco (1600): 70% | decoration neon (140): 70% | néon déco: 50%
)

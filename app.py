import re
import io
import unicodedata
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SEO Clustering App", page_icon="🔗", layout="wide")
st.title("🔗 SEO Clustering (SERP Similarity) → 📈 Ranking decisions")

st.write(
    "1) Dépose ton fichier mots-clés → dédup (sans accents) → clusters\n"
    "2) Dépose ton fichier de **rankings** → l’app décide : **Optimize existing page** (pos 1–20) ou **Create new page** (>20 ou aucune donnée)"
)

# ---------- Utils ----------
def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def normalize_kw(text: str) -> str:
    base = strip_accents(text).lower().strip()
    base = re.sub(r"\s+", " ", base)
    return base

def _read_any(file):
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

# ---------- Similarity parsing ----------
@st.cache_data(show_spinner=False)
def parse_similarity_cell(cell: str):
    if pd.isna(cell) or not str(cell).strip():
        return []
    parts = [p.strip() for p in str(cell).split("|")]
    results = []
    for p in parts:
        p = re.sub(r"\s*%\s*$", "%", p.strip())
        m = re.match(r"(.+?)(?:\(\s*\d+\s*\))?\s*:\s*([\d.,]+)\s*%", p)
        if m:
            kw = m.group(1).strip()
            pct = m.group(2).replace(",", ".").strip()
            try:
                results.append((kw, float(pct)))
            except:
                pass
            continue
        m2 = re.match(r"(.+?)\s+([\d.,]+)\s*%", p)
        if m2:
            kw = m2.group(1).strip()
            pct = m2.group(2).replace(",", ".").strip()
            try:
                results.append((kw, float(pct)))
            except:
                pass
    return results

# ---------- Union-Find ----------
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

# ---------- Step 1: base prep (dedup w/o accents) ----------
@st.cache_data(show_spinner=True)
def prepare_base(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    kw_col = cols.get("keyword") or list(df.columns)[0]
    vol_col = cols.get("monthly vol.") or cols.get("volume") or list(df.columns)[1]
    sim_col = cols.get("kw list and %") or list(df.columns)[2]

    data = df.copy()
    data[kw_col] = data[kw_col].astype(str).str.strip()
    data[vol_col] = pd.to_numeric(data[vol_col], errors="coerce").fillna(0).astype(float)

    file_kw_count_raw = int(data[kw_col].nunique())
    file_total_volume = int(data[vol_col].sum())

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

# ---------- Step 1b: clustering ----------
@st.cache_data(show_spinner=True)
def clusterize(data: pd.DataFrame, kw_col: str, vol_col: str, sim_col: str,
               threshold: float, norm_to_canonical: dict):
    volumes = dict(zip(data[kw_col], data[vol_col]))
    all_kws = list(data[kw_col])
    uf = UnionFind()
    for kw in all_kws:
        uf.add(kw)

    def resolve_canonical(name: str):
        norm = normalize_kw(name)
        canon = norm_to_canonical.get(norm)
        return canon if canon in volumes else None

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

    groups = {}
    for kw in all_kws:
        root = uf.find(kw)
        groups.setdefault(root, []).append(kw)

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

# ---------- Step 2: ranking matching & decision ----------
@st.cache_data(show_spinner=True)
def prepare_ranking(df_rank: pd.DataFrame):
    """
    Rend un DF ranking normalisé.
    Colonnes attendues (noms souples):
      - keyword (obligatoire)
      - url (obligatoire)
      - position (obligatoire, nombre)
    Si d'autres colonnes existent (moteur, pays, device), on les ignore.
    """
    cols = {c.lower(): c for c in df_rank.columns}
    kwc = cols.get("keyword") or cols.get("kw") or list(df_rank.columns)[0]
    urlc = cols.get("url") or list(df_rank.columns)[1]
    posc = cols.get("position") or cols.get("rank") or list(df_rank.columns)[2]

    r = df_rank.copy()
    r[kwc] = r[kwc].astype(str).str.strip()
    r[urlc] = r[urlc].astype(str).str.strip()
    r[posc] = pd.to_numeric(r[posc], errors="coerce")

    r = r.dropna(subset=[kwc, urlc, posc])
    r["__norm_kw"] = r[kwc].map(normalize_kw)

    # pour chaque keyword normalisé, on garde toutes les lignes mais on pourra extraire la best pos
    return r[[kwc, urlc, posc, "__norm_kw"]].rename(
        columns={kwc: "rank_keyword", urlc: "rank_url", posc: "rank_position"}
    )

@st.cache_data(show_spinner=True)
def decide_from_ranking(clusters_df: pd.DataFrame, ranking_norm_df: pd.DataFrame):
    """
    Mappe chaque main_keyword -> ranking rows (via normalisation sans accents).
    Décision:
      - best position in [1,20]  => Optimize existing page
      - else (>20 or no data)    => Create new page
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
                # garder jusqu'à 3 evidences
                evidence_list = [f"{int(p)} — {u}" for p, u in matches_sorted[:3] if pd.notna(p)]

        if best_pos is not None and 1 <= best_pos <= 20:
            decision = "Optimize existing page"
        elif best_pos is not None and best_pos > 20:
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

    # petits compteurs utiles
    n_opt = int((out["decision"] == "Optimize existing page").sum())
    n_create = int((out["decision"].str.startswith("Create new page")).sum())

    return out, n_opt, n_create

# ================= UI =================
with st.sidebar:
    st.markdown("### ℹ️ Tips")
    st.markdown("- Dédup = accents retirés + minuscule, conserve le **plus gros volume**.")
    st.markdown("- Similarité: un sens suffit (A→B ≥ seuil).")
    st.markdown("- Decision ranking: **1–20 = Optimize**, sinon **Create new page**.")

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
    df = _read_any(file)
    (
        data, kw_col, vol_col, sim_col,
        file_kw_count_raw, file_total_volume,
        removed_dups, norm_to_canonical
    ) = prepare_base(df)

    with st.expander("Aperçu (10 premières lignes **après dédup**)"):
        st.dataframe(data.head(10), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Mots-clés (fichier RAW)", value=f"{file_kw_count_raw}")
    with m2:
        st.metric("Volume total (fichier RAW)", value=f"{file_total_volume:,}".replace(",", " "))
    with m3:
        st.metric("Doublons supprimés (sans accents)", value=f"{removed_dups}")

    # Clusters
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
    st.caption("Colonnes attendues (souples) : **Keyword**, **URL**, **Position**. CSV/XLSX accepté.")
    rank_file = st.file_uploader("Ranking file", type=["csv", "xlsx"], key="rankfile")

    if rank_file:
        rank_df_raw = _read_any(rank_file)
        rank_df = prepare_ranking(rank_df_raw)

        with st.expander("Aperçu ranking (top 15)"):
            st.dataframe(rank_df.head(15), use_container_width=True)

        decisions_df, n_opt, n_create = decide_from_ranking(clusters_df, rank_df)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("✅ Optimize existing page (1–20)", n_opt)
        with c2:
            st.metric("🆕 Create new page (no/ >20)", n_create)

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

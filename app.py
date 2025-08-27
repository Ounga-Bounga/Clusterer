import re
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SEO Clustering App", page_icon="🔗", layout="wide")
st.title("🔗 SEO Clustering (SERP Similarity)")
st.write("Dépose un CSV **ou** un XLSX → l’app crée des clusters selon un seuil de similarité (en %).")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def parse_similarity_cell(cell: str):
    """
    Transforme une cellule 'KW list and %' en liste [(keyword, percent_float), ...]
    Gère par ex.:
      'kw (1600): 40%'  'kw: 40 %'  'kw 40%'  séparés par '|'
    """
    if pd.isna(cell) or not str(cell).strip():
        return []
    parts = [p.strip() for p in str(cell).split("|")]
    results = []
    for p in parts:
        # Normaliser espaces autour de %
        p = re.sub(r"\s*%\s*$", "%", p.strip())
        # 1) Forme "kw (123): 40%"
        m = re.match(r"(.+?)(?:\(\s*\d+\s*\))?\s*:\s*([\d.,]+)\s*%", p)
        if m:
            kw = m.group(1).strip()
            pct = m.group(2).replace(",", ".").strip()
            try:
                results.append((kw, float(pct)))
            except:
                pass
            continue
        # 2) Forme fallback "kw 40%"
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

@st.cache_data(show_spinner=True)
def cluster_keywords(df: pd.DataFrame, threshold: float):
    """
    Crée les clusters avec Union-Find.
    On relie A à B si la similarité A→B >= threshold (un seul sens suffit).
    Main keyword = celui avec le volume max du cluster.
    """
    # Standardiser colonnes (ton export est déjà OK)
    cols = {c.lower(): c for c in df.columns}
    kw_col = cols.get("keyword") or list(df.columns)[0]
    vol_col = cols.get("monthly vol.") or cols.get("volume") or list(df.columns)[1]
    sim_col = cols.get("kw list and %") or list(df.columns)[2]

    data = df.copy()
    data[kw_col] = data[kw_col].astype(str).str.strip()
    data[vol_col] = pd.to_numeric(data[vol_col], errors="coerce").fillna(0).astype(float)

    all_kws = data[kw_col].tolist()
    volumes = dict(zip(data[kw_col], data[vol_col]))

    uf = UnionFind()
    for kw in all_kws:
        uf.add(kw)

    for _, row in data.iterrows():
        a = row[kw_col]
        sims = parse_similarity_cell(row.get(sim_col, ""))
        for b, pct in sims:
            b = str(b).strip()
            if b in volumes and pct >= threshold:
                uf.add(b)
                uf.union(a, b)

    groups = {}
    for kw in all_kws:
        root = uf.find(kw)
        groups.setdefault(root, []).append(kw)

    cluster_rows = []
    for i, (root, members) in enumerate(groups.items(), start=1):
        members_sorted = sorted(members, key=lambda k: volumes.get(k, 0), reverse=True)
        main_kw = members_sorted[0]
        total_vol = sum(volumes.get(k, 0) for k in members_sorted)
        cluster_rows.append({
            "cluster_id": i,
            "main_keyword": main_kw,
            "size": len(members_sorted),
            "total_volume": int(total_vol),
            "keywords": ", ".join(members_sorted),
        })

    clusters_df = pd.DataFrame(cluster_rows).sort_values(
        ["size", "total_volume"], ascending=[False, False]
    ).reset_index(drop=True)

    exploded = []
    for _, r in clusters_df.iterrows():
        cid = r["cluster_id"]
        for k in [x.strip() for x in r["keywords"].split(",") if x.strip()]:
            exploded.append({
                "cluster_id": cid,
                "main_keyword": r["main_keyword"],
                "keyword": k,
                "volume": int(volumes.get(k, 0)),
            })
    exploded_df = pd.DataFrame(exploded).sort_values(
        ["cluster_id", "volume"], ascending=[True, False]
    )

    return clusters_df, exploded_df

def to_excel_bytes(sheets: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, d in sheets.items():
            d.to_excel(writer, index=False, sheet_name=name)
    return output.getvalue()

# ---------- UI ----------
left, right = st.columns([1, 2])

with left:
    st.subheader("1) Dépose ton fichier")
    file = st.file_uploader("CSV ou XLSX avec: Keyword | Monthly vol. | KW list and %", type=["csv", "xlsx"])
    threshold = st.slider("Seuil de similarité (%)", min_value=0, max_value=100, value=30, step=5)
    st.caption("Astuce: 30–40% marche bien pour des SERP FR.")

def _read_any(file):
    if file.name.lower().endswith(".xlsx"):
        # lit le premier sheet
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

if file:
    df = _read_any(file)
    with st.expander("Aperçu des 10 premières lignes"):
        st.dataframe(df.head(10), use_container_width=True)

    clusters_df, exploded_df = cluster_keywords(df, float(threshold))

    with right:
        st.subheader("2) Résultats des clusters")
        st.write("**Résumé par cluster**")
        st.dataframe(clusters_df, use_container_width=True, height=360)

        st.write("**Vue détaillée (1 ligne = 1 mot-clé)**")
        st.dataframe(exploded_df, use_container_width=True, height=420)

        xlsx_bytes = to_excel_bytes({"Clusters": clusters_df, "Keywords": exploded_df})
        st.download_button(
            label="⬇️ Télécharger résultats (Excel .xlsx)",
            data=xlsx_bytes,
            file_name="seo_clusters.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        csv_bytes = clusters_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger clusters (CSV)",
            data=csv_bytes,
            file_name="seo_clusters_summary.csv",
            mime="text/csv",
        )
else:
    st.info("Dépose un CSV/XLSX pour lancer le clustering.")

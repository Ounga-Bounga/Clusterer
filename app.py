import re
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SEO Clustering App", page_icon="🔗", layout="wide")
st.title("🔗 SEO Clustering (SERP Similarity)")
st.write("Dépose un CSV → l’app crée des clusters selon un seuil de similarité (en %).")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def parse_similarity_cell(cell: str):
    """
    Parse 'KW list and %' cell into list of tuples: [(keyword, percent_float), ...]
    Accepts formats like:
      'kw: 70% | kw2 (1600): 40% | autre kw: 25%'
    """
    if pd.isna(cell) or not str(cell).strip():
        return []
    parts = [p.strip() for p in str(cell).split("|")]
    results = []
    for p in parts:
        # Try to capture: keyword (optional vol) : percent%
        # Examples matched:
        # "deco neon (1600): 70%", "decoration neon: 40%", "néon déco : 25 %"
        m = re.match(r"(.+?)(?:\(\s*\d+\s*\))?\s*:\s*([\d.,]+)\s*%", p)
        if m:
            kw = m.group(1).strip()
            pct = m.group(2).replace(",", ".").strip()
            try:
                results.append((kw, float(pct)))
            except:
                pass
        else:
            # Fallback: try "keyword 70%" (without colon)
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
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
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
    Build clusters using Union-Find.
    Edge between A and B if A->B similarity >= threshold (one-way is enough).
    Main keyword per cluster = max volume.
    """
    # Standardize columns
    cols = {c.lower(): c for c in df.columns}
    kw_col = cols.get("keyword") or list(df.columns)[0]
    vol_col = cols.get("monthly vol.") or cols.get("volume") or list(df.columns)[1]
    sim_col = cols.get("kw list and %") or list(df.columns)[2]

    # Clean base dataframe
    data = df.copy()
    data[kw_col] = data[kw_col].astype(str).str.strip()
    # Ensure numeric volume
    data[vol_col] = pd.to_numeric(data[vol_col], errors="coerce").fillna(0).astype(float)

    # Map for quick lookup
    all_kws = data[kw_col].tolist()
    volumes = dict(zip(data[kw_col], data[vol_col]))

    uf = UnionFind()
    for kw in all_kws:
        uf.add(kw)

    # Build edges using threshold
    for _, row in data.iterrows():
        a = row[kw_col]
        sims = parse_similarity_cell(row.get(sim_col, ""))
        for b, pct in sims:
            b = str(b).strip()
            if b in volumes and pct >= threshold:
                uf.add(b)
                uf.union(a, b)

    # Group by root
    groups = {}
    for kw in all_kws:
        root = uf.find(kw)
        groups.setdefault(root, []).append(kw)

    # Build clusters summary
    cluster_rows = []
    cluster_id = 1
    for root, members in groups.items():
        # sort members by volume desc for readability
        members_sorted = sorted(members, key=lambda k: volumes.get(k, 0), reverse=True)
        main_kw = members_sorted[0]
        total_vol = sum(volumes.get(k, 0) for k in members_sorted)
        cluster_rows.append({
            "cluster_id": cluster_id,
            "main_keyword": main_kw,
            "size": len(members_sorted),
            "total_volume": int(total_vol),
            "keywords": ", ".join(members_sorted),
        })
        cluster_id += 1

    clusters_df = pd.DataFrame(cluster_rows).sort_values(
        ["size", "total_volume"], ascending=[False, False]
    ).reset_index(drop=True)

    # Exploded view (one row per keyword, with its cluster)
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
    """
    sheets: {"Clusters": df1, "Keywords": df2}
    returns bytes of an .xlsx file
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, d in sheets.items():
            d.to_excel(writer, index=False, sheet_name=name)
    return output.getvalue()

# ---------- UI ----------
left, right = st.columns([1, 2])

with left:
    st.subheader("1) Dépose ton CSV")
    file = st.file_uploader("CSV avec: Keyword | Monthly vol. | KW list and %", type=["csv"])
    threshold = st.slider("Seuil de similarité (%)", min_value=0, max_value=100, value=30, step=5)
    st.caption("Astuce: 30–40% marche bien pour des SERP FR.")

if file:
    df = pd.read_csv(file)
    with st.expander("Aperçu des 10 premières lignes"):
        st.dataframe(df.head(10), use_container_width=True)

    clusters_df, exploded_df = cluster_keywords(df, float(threshold))

    with right:
        st.subheader("2) Résultats des clusters")
        st.write("**Résumé par cluster**")
        st.dataframe(clusters_df, use_container_width=True, height=360)

        st.write("**Vue détaillée (1 ligne = 1 mot-clé)**")
        st.dataframe(exploded_df, use_container_width=True, height=420)

        # Downloads
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
    st.info("Dépose un CSV pour lancer le clustering.")

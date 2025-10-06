import io
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

st.set_page_config(page_title="Shipping Data Grapher", layout="wide")

# Top row with logo and title
col1, col2 = st.columns([1, 4])

with col1:
    st.image("winthrop_inverted.png", width=600)  # replace with your logo file

with col2:
    st.markdown(
        "<h1 style='color:black; font-family:sans-serif; margin-bottom:0;'>"
        "Shipping Data Dashboard"
        "</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:gray; font-size:16px; margin-top:0;'>"
        "Explore shipping performance trends by product"
        "</p>",
        unsafe_allow_html=True
    )

st.title("Shipping Data Grapher")
st.write("Upload your shipping CSV and explore trends.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")

    # Period selection mapped to pandas offset aliases
    period_label = st.selectbox(
        "Period of Summation",
        options=["Weekly", "Monthly", "Quarterly", "Yearly"],
        index=0
    )
    PERIOD_TO_FREQ = {
        "Weekly": "W",        # weekly, end of week
        "Monthly": "M",       # month end
        "Quarterly": "Q",     # quarter end (calendar)
        "Yearly": "Y",        # year end (calendar)
    }
    period_code = PERIOD_TO_FREQ[period_label]

    last_n_months = st.number_input("Window for Table of Averages (months)", min_value=1, max_value=120, value=6)
    show_total_series = st.checkbox("Show Combined Total Series", value=True)

    st.markdown("---")
    st.header("Choose Products to Make Individual Graphs")

    # 🔹 Hard-coded product substrings (edit these lists as needed)
    productfamilies = ["Butts", "Tops", "Guides", "Gaffs"]
    individualproducts = ["Terminator", "Epic", "T-10X", "(EX)", "(XP)", "(ER)", "(XAT)", "(AT)", "(TT)", "Aussie"]
    colors = ["Black", "Silver", "Gold", "Blue", "Custom"]
    termhandlen = ["Short", "Long"]
    gimballen = ["Short", "Long"]

    # --- Product Families ---
    st.subheader("Product Families")
    selected_families = []
    for prod in productfamilies:
        if st.checkbox(prod, key=f"fam_{prod}"):
            selected_families.append(prod)

    # --- Individual Products ---
    st.subheader("Individual Products")
    selected_individual = []
    for prod1 in individualproducts:
        if st.checkbox(prod1, key=f"ind_{prod1}"):
            selected_individual.append(prod1)

    # --- Colours ---
    st.subheader("Colours")
    selected_colors = []
    for color in colors:
        if st.checkbox(color, key=f"col_{color}"):
            selected_colors.append(color)

    # --- Terminator Handle Length ---
    st.subheader("Terminator Handle Length")
    selected_term_len = []
    for term in termhandlen:
        # label is "Short Handle" / "Long Handle", but we only store "Short"/"Long"
        if st.checkbox(f"{term} Handle", key=f"term_{term}"):
            selected_term_len.append(term)

    # --- Gimbal Length ---
    st.subheader("Gimbal Length")
    selected_gimbal_len = []
    for gimbal in gimballen:
        if st.checkbox(f"{gimbal} Gimbal", key=f"gim_{gimbal}"):
            selected_gimbal_len.append(gimbal)

    # Combined list of patterns used for per-product charts
    patterns = (
        selected_families
        + selected_individual
        + selected_colors
        + selected_term_len
        + selected_gimbal_len
    )

    st.markdown("---")
    st.subheader("Choose Products to Sum Totals in a Graph")

    reuse_for_totals = st.checkbox("Use selections above for totals", value=True)
    if reuse_for_totals:
        total_patterns = patterns
    else:
        st.caption("Pick separate items for the combined total series:")
        sel_fam_total, sel_ind_total, sel_col_total, sel_term_total, sel_gim_total = [], [], [], [], []

        st.text("Families")
        for prod in productfamilies:
            if st.checkbox(f"[Totals] {prod}", key=f"tot_fam_{prod}"):
                sel_fam_total.append(prod)

        st.text("Individual")
        for prod1 in individualproducts:
            if st.checkbox(f"[Totals] {prod1}", key=f"tot_ind_{prod1}"):
                sel_ind_total.append(prod1)

        st.text("Colours")
        for color in colors:
            if st.checkbox(f"[Totals] {color}", key=f"tot_col_{color}"):
                sel_col_total.append(color)

        st.text("Terminator Handle Length")
        for term in termhandlen:
            if st.checkbox(f"[Totals] Handle {term}", key=f"tot_term_{term}"):
                sel_term_total.append(term)

        st.text("Gimbal Length")
        for gimbal in gimballen:
            if st.checkbox(f"[Totals] Gimbal {gimbal}", key=f"tot_gim_{gimbal}"):
                sel_gim_total.append(gimbal)

        total_patterns = sel_fam_total + sel_ind_total + sel_col_total + sel_term_total + sel_gim_total

    st.markdown("---")
    st.write("**Selected (individual graphs):**", patterns if patterns else "None")
    st.write("**Selected (totals):**", total_patterns if total_patterns else "None")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# ---------- Helpers ----------

# === Custom include/exclude logic for families + individual products + handle length ===
FAMILY_INCLUDES = {
    "tops":   ["XAT", "AUSSIE", "AT", "TT"],
    "guides": ["XC", "XP", "ER", "ES"],
    "butts":  ["EPIC", "TRMTR", "T-10X"],
}
INDIVIDUAL_INCLUDES = {
    "terminator": ["TRMTR"],
    "epic":       ["EPIC"],
    "t-10x":      ["T-10X"],
}
# Exclusions always win
EXCLUDE_PRIORITY = ["AC", "CART", "SET", "SFC", "SA"]

def _token_match(s: pd.Series, tokens: list[str]) -> pd.Series:
    if not tokens:
        return pd.Series(False, index=s.index)
    pat = "|".join(re.escape(t) for t in tokens)
    return s.str.contains(pat, case=False, na=False, regex=True)

def _exclude_match(s: pd.Series) -> pd.Series:
    if not EXCLUDE_PRIORITY:
        return pd.Series(False, index=s.index)
    pat_exc = "|".join(re.escape(t) for t in EXCLUDE_PRIORITY)
    return s.str.contains(pat_exc, case=False, na=False, regex=True)

def get_mask_for_pattern(df: pd.DataFrame, pat: str, selected_labels: set[str]) -> pd.Series:
    """
    Returns a boolean mask for checkbox label `pat`, using `selected_labels`
    to enforce dependencies (e.g., handle length only active if Terminator or Butts selected).
    """
    col = "Part Number"
    s = df[col].astype(str)
    key = pat.lower()
    selected_lc = {x.lower() for x in selected_labels}

    # Families (custom includes + exclusions)
    if key in FAMILY_INCLUDES:
        m_inc = _token_match(s, FAMILY_INCLUDES[key])
        m_exc = _exclude_match(s)
        return m_inc & ~m_exc

    # Individuals (custom includes + exclusions)
    if key in INDIVIDUAL_INCLUDES:
        m_inc = _token_match(s, INDIVIDUAL_INCLUDES[key])
        m_exc = _exclude_match(s)
        return m_inc & ~m_exc

    # Handle length (dependency: Terminator OR Butts selected)
    # Accept "handle short/long" OR just "short"/"long"
    if key in {"handle short", "handle long", "short", "long"}:
        depends_on = {"terminator", "butts"}
        if selected_lc.isdisjoint(depends_on):
            # Neither Terminator nor Butts selected → no effect
            return pd.Series(False, index=s.index)

        # Only consider TRMTR entries, then split by SHT/LNG
        base = _token_match(s, ["TRMTR"])
        if key in {"handle short", "short"}:
            spec = s.str.contains(r"SHT", case=False, na=False, regex=True)
        else:  # long
            spec = s.str.contains(r"LNG", case=False, na=False, regex=True)

        m_exc = _exclude_match(s)
        return base & spec & ~m_exc

    # Fallback: generic case-insensitive substring of the label itself
    return s.str.contains(re.escape(pat), case=False, na=False, regex=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: Ship Date, Part Number, Quantity To Ship
    df = df.copy()
    col_map = {c.lower(): c for c in df.columns}
    for target in ["Ship Date", "Part Number", "Quantity To Ship"]:
        match = next((orig for low, orig in col_map.items()
                      if low.replace(" ", "") == target.lower().replace(" ", "")), None)
        if match and match != target:
            df.rename(columns={match: target}, inplace=True)

    if "Ship Date" not in df.columns:
        raise ValueError("Missing 'Ship Date' column.")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df = df.dropna(subset=["Ship Date"])

    if "Quantity To Ship" not in df.columns:
        raise ValueError("Missing 'Quantity To Ship' column.")
    df["Quantity To Ship"] = pd.to_numeric(df["Quantity To Ship"], errors="coerce").fillna(0)

    if "Part Number" not in df.columns:
        raise ValueError("Missing 'Part Number' column.")
    return df

def period_sum(df_filtered: pd.DataFrame, freq_code: str) -> pd.Series:
    return (
        df_filtered
        .groupby(pd.Grouper(key="Ship Date", freq=freq_code))["Quantity To Ship"]
        .sum()
        .sort_index()
    )

def plot_series(ax, series: pd.Series, title: str, ylabel: str, show_trend=True, show_avg=True):
    if series.empty:
        ax.text(0.5, 0.5, "No data for selection", ha="center", va="center")
        ax.set_axis_off()
        return
    x = np.arange(len(series))
    y = series.values
    ax.plot(series.index, y, marker='o', linestyle='-', label='Total')
    if show_avg:
        avg = float(np.mean(y))
        ax.axhline(avg, linestyle=':', label=f'Average ({avg:.2f})')
    if show_trend and len(y) >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        trend = np.poly1d(coeffs)
        ax.plot(series.index, trend(x), linestyle='--', label='Trend')
    ax.set_title(title)
    ax.set_xlabel("Period")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

def build_pdf(df: pd.DataFrame, patterns: list[str], freq_code: str,
              last_n_months: int, period_label: str, end_date, selected_labels: set[str]) -> bytes:
    # Prepare series per product using the SAME logic as the app
    series_by_product = {}
    for pat in patterns:
        mask = get_mask_for_pattern(df, pat, selected_labels)
        s = period_sum(df[mask], freq_code)
        series_by_product[pat] = s

    # Total as combined
    if patterns:
        masks = [get_mask_for_pattern(df, p, selected_labels) for p in patterns]
        combined_mask = np.logical_or.reduce(masks) if masks else pd.Series(False, index=df.index)
        combined = df[combined_mask]
    else:
        combined = df
    total_series = period_sum(combined, freq_code)

    # Average window aligned to UI end_date
    cutoff = pd.Timestamp(end_date) - pd.DateOffset(months=last_n_months)
    avg_last_window = {}
    for pat, s in series_by_product.items():
        last = s[s.index >= cutoff]
        avg_last_window[pat] = float(last.mean()) if not last.empty else 0.0

    # Build PDF into memory
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover
        fig_cover, ax_cover = plt.subplots(figsize=(11, 8.5))
        ax_cover.axis('off')
        date_range = f"{df['Ship Date'].min().date()} to {df['Ship Date'].max().date()}"
        ax_cover.text(0.5, 0.7, "Shipment Report", ha='center', va='center', fontsize=24)
        ax_cover.text(0.5, 0.6, f"{period_label} totals and last window averages", ha='center', va='center', fontsize=14)
        ax_cover.text(0.5, 0.52, date_range, ha='center', va='center', fontsize=12)
        pdf.savefig(fig_cover); plt.close(fig_cover)

        # Product charts
        for pat, s in series_by_product.items():
            fig, ax = plt.subplots(figsize=(11, 6))
            plot_series(ax, s, f"{period_label} {pat} Shipped", "Quantity Shipped")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Total chart
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_series(ax, total_series, f"{period_label} Total Quantity Shipped (All Selected Products)", "Quantity Shipped")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Averages table
        avg_df = pd.DataFrame({
            "Product": list(avg_last_window.keys()),
            f"Avg per {period_label} (Last {last_n_months} mo)": list(avg_last_window.values())
        })
        # sort descending by the average column
        avg_col = avg_df.columns[-1]
        avg_df = avg_df.sort_values(by=avg_col, ascending=False)

        fig_tbl, ax_tbl = plt.subplots(figsize=(11, 4))
        ax_tbl.axis('off')
        the_table = ax_tbl.table(cellText=avg_df.values, colLabels=avg_df.columns, loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1, 1.2)
        fig_tbl.tight_layout()
        pdf.savefig(fig_tbl); plt.close(fig_tbl)

    buf.seek(0)
    return buf.getvalue()

# ---------- Main flow ----------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    try:
        df = normalize(df)
    except Exception as e:
        st.error(f"Error: {e}")
    else:
        # Date filter
        min_date = df["Ship Date"].min().date()
        max_date = df["Ship Date"].max().date()
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
        with c2:
            end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

        mask = (df["Ship Date"].dt.date >= start_date) & (df["Ship Date"].dt.date <= end_date)
        df_filtered = df.loc[mask].copy()

        st.subheader(f"{period_label} charts")

        # Compose a lowercase set of currently selected labels for dependency logic
        current_selection = set(patterns)

        # Individual product charts
        for pat in patterns:
            subset = df_filtered[get_mask_for_pattern(df_filtered, pat, current_selection)]
            s = period_sum(subset, period_code)
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_series(ax, s, f"{pat} — {period_label} shipped", "Quantity Shipped")
            st.pyplot(fig)

        # Combined total series (using either same or separate selections)
        if show_total_series:
            if total_patterns:
                totals_selection = set(total_patterns)
                masks = [get_mask_for_pattern(df_filtered, p, totals_selection) for p in total_patterns]
                mask_total = np.logical_or.reduce(masks) if masks else pd.Series(False, index=df_filtered.index)
                combined = df_filtered[mask_total]
            else:
                combined = df_filtered

            total_s = period_sum(combined, period_code)
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_series(ax, total_s, f"All selected — {period_label} shipped", "Quantity Shipped")
            st.pyplot(fig)

        # Averages table (last N months from end_date)
        cutoff = pd.Timestamp(end_date) - pd.DateOffset(months=last_n_months)
        avgs = []
        for pat in patterns:
            subset = df_filtered[get_mask_for_pattern(df_filtered, pat, current_selection)]
            s = period_sum(subset, period_code)
            last = s[s.index >= cutoff]
            avgs.append({
                "Product": pat,
                f"Avg per {period_label} (Last {last_n_months} mo)": float(last.mean()) if not last.empty else 0.0
            })
        avg_df = pd.DataFrame(avgs)
        st.subheader("Averages")
        st.dataframe(avg_df)

        # PDF download
        st.subheader("Export")
        if st.button("Build PDF report"):
            pdf_bytes = build_pdf(
                df_filtered, patterns, period_code, last_n_months, period_label, end_date, set(patterns)
            )
            st.download_button(
                "Download Shipment_Report.pdf",
                data=pdf_bytes,
                file_name="Shipment_Report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Upload a CSV to get started. Expected columns: 'Ship Date', 'Part Number', 'Quantity To Ship'.")

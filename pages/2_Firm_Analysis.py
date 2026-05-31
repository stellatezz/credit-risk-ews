import streamlit as st
import pandas as pd
from src.ews.config import FIRMS

st.set_page_config(page_title="Firm Analysis", layout="wide")

st.title("🏢 Firm Analysis")
st.markdown("Drill down into individual firm risk profiles and feature trends")

st.markdown("---")

# Firm selector
firm_options = sorted(FIRMS.keys())
selected_firm = st.selectbox("Select a Firm:", firm_options)

if selected_firm:
    firm_name = FIRMS[selected_firm]['name']
    firm_industry = FIRMS[selected_firm]['industry']

    st.header(f"{selected_firm} — {firm_name}")
    st.subheader(f"📊 {firm_industry} Sector")

    st.markdown("---")

    try:
        # Load the panel
        panel = pd.read_csv("data/processed/panel_phase2.csv")
        firm_data = panel[panel['ticker'] == selected_firm].copy()

        if len(firm_data) == 0:
            st.warning(f"⚠️ No data available for {selected_firm} (possibly delisted post-bankruptcy)")
        else:
            firm_data['date'] = pd.to_datetime(firm_data['date'])
            firm_data = firm_data.sort_values('date')

            # Key metrics
            st.header("📈 Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            events = firm_data['label_a'].sum()
            total_periods = len(firm_data)
            event_rate = firm_data['label_a'].mean()
            date_range = f"{firm_data['date'].min().strftime('%Y-%m')} to {firm_data['date'].max().strftime('%Y-%m')}"

            with col1:
                st.metric("Total Distress Events", int(events))
            with col2:
                st.metric("Periods Covered", total_periods)
            with col3:
                st.metric("Event Rate", f"{event_rate:.1%}")
            with col4:
                st.metric("Date Range", date_range)

            st.markdown("---")

            # Risk timeline
            st.header("1️⃣ Distress Indicator Over Time")
            st.write("""
            Binary indicator: 1 = firm experienced ≥40% equity drawdown in the next 12 months.
            This is the actual outcome your model tries to predict.
            """)

            st.line_chart(firm_data.set_index('date')['label_a'], height=300)

            st.markdown("---")

            # Feature trends
            st.header("2️⃣ Market Features Over Time")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Returns (1m, 3m, 6m)")
                st.write("Monthly cumulative equity returns. Negative returns often precede distress.")
                features_ret = ['ret_1m', 'ret_3m', 'ret_6m']
                available_ret = [f for f in features_ret if f in firm_data.columns]
                if available_ret:
                    st.line_chart(firm_data.set_index('date')[available_ret], height=300)

            with col2:
                st.subheader("Volatility (3m, 6m)")
                st.write("Annualized realized volatility. Spikes often coincide with distress episodes.")
                features_vol = ['vol_3m', 'vol_6m']
                available_vol = [f for f in features_vol if f in firm_data.columns]
                if available_vol:
                    st.line_chart(firm_data.set_index('date')[available_vol], height=300)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("12-Month Drawdown")
                st.write("Peak-to-trough decline over prior 252 trading days. Direct measure of distress magnitude.")
                if 'drawdown_12m' in firm_data.columns:
                    st.line_chart(firm_data.set_index('date')[['drawdown_12m']], height=300)

            with col2:
                st.subheader("VIX (Macro regime)")
                st.write("CBOE volatility index. High VIX = stressed market environment.")
                if 'vix' in firm_data.columns:
                    st.line_chart(firm_data.set_index('date')[['vix']], height=300)

            st.markdown("---")

            st.header("3️⃣ Accounting Features Over Time")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Leverage (Total Liab / Total Assets)")
                st.write("Higher leverage = more financial risk. Deteriorating firms often see rising leverage.")
                if 'leverage' in firm_data.columns:
                    st.line_chart(firm_data.set_index('date')[['leverage']], height=300)

            with col2:
                st.subheader("Liquidity Buffer (Cash / Total Assets)")
                st.write("Cushion to handle shocks. Lower liquidity = higher distress risk.")
                if 'liquidity_buffer' in firm_data.columns:
                    st.line_chart(firm_data.set_index('date')[['liquidity_buffer']], height=300)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Working Capital Ratio")
                st.write("(Current Assets - Current Liabilities) / Total Assets. Indicates operational health.")
                if 'wc_ratio' in firm_data.columns:
                    st.line_chart(firm_data.set_index('date')[['wc_ratio']], height=300)

            with col2:
                st.subheader("Profitability (EBIT / Total Assets)")
                st.write("Margin on total asset base. Negative profitability is a red flag.")
                if 'profitability' in firm_data.columns:
                    st.line_chart(firm_data.set_index('date')[['profitability']], height=300)

            st.markdown("---")

            st.header("💡 Key Observations for {0}".format(selected_firm))

            distress_periods = firm_data[firm_data['label_a'] == 1].copy()
            if len(distress_periods) > 0:
                st.write(f"**Distress records:** {len(distress_periods)}")
                st.write("**Months with distress indicator:**")
                # Format for readable table
                display_df = distress_periods[['date', 'label_a', 'ret_6m', 'drawdown_12m', 'leverage']].copy()
                display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m')
                display_df = display_df.rename(columns={
                    'date': 'Month',
                    'label_a': 'Distress',
                    'ret_6m': 'Ret 6m',
                    'drawdown_12m': 'Drawdown 12m',
                    'leverage': 'Leverage'
                })
                st.table(display_df.reset_index(drop=True))
            else:
                st.write("✅ No distress events recorded in this period.")

            st.markdown("---")

            st.info("""
            **Interpretation Tips:**
            - Match visual peaks in returns/volatility/drawdowns with distress events
            - Look for declining liquidity + rising leverage = warning sign
            - Negative profitability + high volatility = compounding risk
            - Use this view to educate credit analysts on which firms to monitor closely
            """)

    except Exception as e:
        st.error(f"Could not load firm data: {e}")


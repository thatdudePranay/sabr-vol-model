import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sabr import SABR, hagan_sabr_vol
from datetime import datetime

#To be updated frontend

st.set_page_config(
    page_title="SABR Volatility Model",
    page_icon="farmsa.png",
    layout = "wide"
)

st.title("SABR Volatility Model")
st.markdown("Stochastic Alpha Beta Rho (SABR) model is a popular model used in finance to capture the dynamics of implied volatility surfaces. It is widely used for pricing and managing the risk of options and other derivatives. The SABR model is characterized by its ability to capture the skew and smile effects observed in implied volatility surfaces, making it a valuable tool for traders and risk managers.")

st.markdown("Calibrate the SABR model to live options data and analyze implied volatility.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("Ticker Symbol", value="Enter Ticker").upper()

with col2:
    option_type = st.selectbox(
        "Option Type",
        options=["CALLS", "PUTS"],
        index=0
    )

# Show the actual expiry date so user knows what they selected
if ticker:
    try:
        stock = yf.Ticker(ticker)
        expiries = list(stock.options)

        if expiries:
            expiry_index = st.slider(
                "Expiry Date",
                min_value=0,
                max_value=len(expiries) - 1,
                value=min(5, len(expiries) - 1),
            )
            selected_expiry = expiries[expiry_index]
            days_out = (datetime.strptime(selected_expiry, '%Y-%m-%d') - datetime.now()).days
            st.caption(f"Selected: **{selected_expiry}** ({days_out} days out)")
        else:
            st.error("No options data found for this ticker.")
            expiry_index = 0

    except Exception as e:
        st.error(f"Could not fetch expiries: {e}")
        expiry_index = 0

st.divider()

run_button = st.button("Run Model", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 2: CALIBRATION
# When user clicks Run Model:
#   - SABR class is instantiated with the ticker
#   - calibrate() fetches live options data from yfinance
#   - fits alpha, rho, nu to the market smile
#   - stores model.strikes and model.market_vols (live yfinance IVs)
#   - model is saved to session_state so it persists
# ─────────────────────────────────────────────

if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching live options data and calibrating SABR for {ticker}..."):
            try:
                # This is where we instantiate and calibrate our SABR model
                # calibrate() internally calls yf.Ticker(ticker).option_chain(expiry)
                # which pulls live bid/ask/IV data from yfinance for that expiry
                model = SABR(ticker, beta=0.5)
                model.calibrate(expiry_index=expiry_index, option_type=option_type)

                # Save to session state so outputs persist without recalibrating
                st.session_state['model'] = model
                st.success(f"✅ SABR model successfully calibrated for {ticker} — enter a strike below to compare IVs")

            except ValueError as e:
                st.error(f"Calibration error: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# SECTION 3: OUTPUTS
# Only shown after successful calibration

if 'model' in st.session_state:
    model = st.session_state['model']

    # Fit quality metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spot Price", f"${model.spot:.2f}")
    col2.metric("Forward Price", f"${model.forward:.2f}")
    col3.metric("RMSE", f"{model.rmse*100:.4f}%")
    col4.metric("Max Error", f"{model.max_error*100:.4f}%")

    st.divider()

    # ── Volatility Smile ──
    # model.strikes and model.market_vols are the raw yfinance IVs
    # fetched during calibrate() — these are the blue market dots
    # model.get_vol(k) calls hagan_sabr_vol() with calibrated params
    # to compute SABR IV at each point — this is the red curve
    st.subheader("Volatility Smile")
    k_range = np.linspace(model.spot * 0.8, model.spot * 1.2, 100)
    sabr_vols = [model.get_vol(k) * 100 for k in k_range]

    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=model.strikes,
        y=model.market_vols * 100,
        mode='markers',
        name='Market IV (yfinance)',
        marker=dict(color='royalblue', size=8)
    ))
    fig_smile.add_trace(go.Scatter(
        x=k_range,
        y=sabr_vols,
        mode='lines',
        name='SABR IV',
        line=dict(color='crimson', width=2)
    ))
    fig_smile.add_vline(x=model.spot, line_dash="dash", line_color="gray",
                        annotation_text="Spot", annotation_position="top")
    fig_smile.update_layout(
        xaxis_title="Strike ($)",
        yaxis_title="Implied Vol (%)",
        template="plotly_dark",
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_smile, use_container_width=True)

    st.divider()

    # ── Volatility Surface ──
    # Uses calibrated SABR params to project IV across
    # all strikes and maturities via hagan_sabr_vol()
    st.subheader("Volatility Surface")
    fig_surface = model.plot_surface()
    fig_surface.update_layout(
        template="plotly_dark",
        height=1000,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.8)
        )
    )
    st.plotly_chart(fig_surface, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────
    # SECTION 4: SABR vs MARKET IV AT STRIKE
    # SABR IV: model.get_vol(strike) calls hagan_sabr_vol()
    #          with our calibrated alpha, rho, nu
    # Market IV: we find the nearest traded strike in
    #            model.strikes (from yfinance) and return
    #            the corresponding model.market_vols value
    # ─────────────────────────────────────────────
    st.subheader("SABR vs Market IV at Strike")
    st.info("Enter a strike price below to compare SABR implied volatility against the live market IV from yfinance.")

    strike_input = st.number_input(
        "Strike Price ($)",
        min_value=float(model.spot * 0.8),
        max_value=float(model.spot * 1.2),
        value=float(model.spot),
        step=1.0
    )

    # SABR IV — computed from our calibrated model
    sabr_iv = model.get_vol(strike_input) * 100

    # Market IV — pulled from yfinance data stored in model.market_vols
    # We find the nearest traded strike since not every strike is traded
    nearest_idx = np.argmin(np.abs(model.strikes - strike_input))
    nearest_strike = model.strikes[nearest_idx]
    market_iv = model.market_vols[nearest_idx] * 100
    diff = sabr_iv - market_iv

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SABR IV", f"{sabr_iv:.2f}%")
    c2.metric("Market IV", f"{market_iv:.2f}%")
    c3.metric("Nearest Traded Strike", f"${nearest_strike:.2f}")
    c4.metric("Difference (SABR - Market)", f"{diff:.2f}%", delta=f"{diff:.2f}%")
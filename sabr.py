import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from scipy.optimize import least_squares
from scipy.stats import norm
import plotly.graph_objects as go


def _compute_x_z_ratio(z, rho):
    """
    Compute z/x(z) with proper handling of edge cases.
    - z ≈ 0 (ATM options)
    - rho = 1 (causes division by zero in original)
    - rho = -1 (causes log of negative number)
    """

    eps = 1e-10
    # if z = 0, limit would be 1
    if abs(z) < eps:
      return 1.0

    #for if/when rho = 1,
    if abs(rho - 1.0) < eps:
      if z > -1:
        return z / np.log(1 + z) if abs(np.log(1 + z)) > eps else 1.0
      return 1.0

    #for if/when rho = -1
    if abs(rho + 1.0) < eps:
      if z < 1:
        return z / np.log(1 - z) if abs(np.log(1 - z)) > eps else 1.0
      return 1.0

    sqrt_term = np.sqrt(1 - 2*rho*z + z**2)
    numerator = sqrt_term + z - rho
    denominator = 1 - rho

    if numerator / denominator <= 0:
      return 1.0

    x_z = np.log(numerator / denominator)

    if abs(x_z) < eps:
      return 1.0

    return z / x_z

def hagan_sabr_vol(k, f, t, alpha, beta, rho, nu):
    """
    Hagan (2002) SABR lognormal implied volatility approximation.
    """
    if alpha <= 0:
        return 1e-6

    # ATM case
    if abs(f - k) < 1e-10:
        fk_beta = f ** (1 - beta)
        term1 = alpha / fk_beta
        term2 = 1 + (
            ((1 - beta)**2 / 24) * (alpha**2 / f**(2 - 2*beta)) +
            (rho * beta * nu * alpha) / (4 * fk_beta) +
            ((2 - 3*rho**2) / 24) * nu**2
        ) * t
        return term1 * term2

    # Non-ATM case
    fk = f * k
    log_fk = np.log(f / k)
    fk_beta_half = fk ** ((1 - beta) / 2)

    z = (nu / alpha) * fk_beta_half * log_fk

    x_z_ratio = _compute_x_z_ratio(z, rho)

    term1 = alpha / (
        fk_beta_half *
        (1 + ((1 - beta)**2 / 24) * log_fk**2 +
         ((1 - beta)**4 / 1920) * log_fk**4)
    )

    term2 = x_z_ratio

    term3 = 1 + (
        ((1 - beta)**2 / 24) * (alpha**2 / fk**(1 - beta)) +
        (rho * beta * nu * alpha) / (4 * fk_beta_half) +
        ((2 - 3*rho**2) / 24) * nu**2
    ) * t

    return term1 * term2 * term3


class SABR:
    """
    SABR model calibrated to yfinance options data.

    Usage:
        model = SABR("SPY")
        model.calibrate()
        print(model.alpha, model.rho, model.nu)
        vol = model.get_vol(500)
    """

    def __init__(self, ticker, beta=0.5):
        self.ticker = ticker.upper()
        self.beta = beta
        self.alpha = None
        self.rho = None
        self.nu = None
        self.rmse = None
        self.max_error = None


    def calibrate(self, expiry_index=0, r=0.05, option_type ='both'):
        """
        Fetch data and calibrate SABR parameters.

        Args:
            expiry_index: Which expiration (0=nearest)
            r: Risk-free rate for forward calculation
            option_type: 'call' or 'put' or 'both'
        """
        # Fetch data
        option_type = option_type.lower()
        stock = yf.Ticker(self.ticker)
        self.spot = stock.history(period='1d')['Close'].iloc[-1]

        expiry = stock.options[expiry_index]
        self.expiry = expiry

        days = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
        self.t = max(days / 365, 1/365)
        self.forward = self.spot * np.exp(r * self.t)
        self.option_type = option_type

        # Get options and filter
        chain = stock.option_chain(expiry)

        if option_type == 'calls':
          df = chain.calls.copy()
        elif option_type == 'puts':
          df = chain.puts.copy()
        elif option_type == 'both':
          calls = chain.calls.copy()
          puts = chain.puts.copy()
          merged = calls.merge(puts, on='strike', suffixes=('_call', '_put'))
          merged = merged[
              (merged['volume_call'] >= 10) &
              (merged['volume_put'] >= 10)
          ]
          merged['impliedVolatility'] = (merged['impliedVolatility_call'] + merged['impliedVolatility_put']) / 2
          merged['volume'] = merged['volume_call'] + merged['volume_put']
          df = merged[['strike', 'impliedVolatility', 'volume']].copy()



        if option_type == 'both':
            df = df[(df['impliedVolatility'] > 0.01) & (df['impliedVolatility'] < 2)]
        elif option_type == 'puts':
            df = df[(df['impliedVolatility'] > 0.01) &
                    (df['impliedVolatility'] < 2) &
                    (df['volume'] >= 5)]
        else:  # calls
            df = df[(df['impliedVolatility'] > 0.01) &
                    (df['impliedVolatility'] < 2) &
                    (df['volume'] >= 10)]

        df = df[(df['strike'] >= self.spot * 0.8) &
                (df['strike'] <= self.spot * 1.2)]
        
        vols = df['impliedVolatility'].values
        median_vol = np.median(vols)
        df = df[np.abs(df['impliedVolatility'] - median_vol) < 3 * np.std(vols)]

        self.strikes = df['strike'].values
        self.market_vols = df['impliedVolatility'].values

        if len(self.strikes) < 3:
            raise ValueError("Not enough liquid options for calibration")

        atm_idx = np.argmin(np.abs(self.strikes - self.forward))
        atm_vol = self.market_vols[atm_idx]
        initial_alpha = atm_vol * self.forward ** (1 - self.beta)

        alpha_upper = max(10.0, initial_alpha * 2)

        def residuals(params):
            alpha, rho, nu = params
            model_vols = np.array([
                hagan_sabr_vol(k, self.forward, self.t, alpha, self.beta, rho, nu)
                for k in self.strikes
            ])
            moneyness = np.abs(np.log(self.strikes / self.forward))
            weights = np.exp(-2 * moneyness)
            return weights * (model_vols - self.market_vols)


        best_result = None
        initial_guesses = [
            [initial_alpha, -0.2, 0.3],
            [initial_alpha, -0.5, 1.0],
            [initial_alpha * 0.5, -0.3, 0.5],
            [initial_alpha * 1.5, -0.1, 2.0],
        ]

        for x0 in initial_guesses:
            try:
                result = least_squares(
                    residuals,
                    x0=x0,
                    bounds=([1e-6, -0.999, 1e-6], [alpha_upper, 0.999, 10.0]),
                    method='trf'
                )
                if best_result is None or result.cost < best_result.cost:
                    best_result = result
            except Exception:
                continue

        result = best_result
        if result is None:
          raise ValueError("Calibration failed. Try a different expiry_index.")

        self.alpha, self.rho, self.nu = result.x

        model_vols = np.array([self.get_vol(k) for k in self.strikes])
        errors = model_vols - self.market_vols
        self.rmse = np.sqrt(np.mean(errors**2))
        self.max_error = np.max(np.abs(errors))

        print(f"{self.ticker} | Spot: ${self.spot:.2f} | Expiry: {expiry} ({days}d)")
        print(f"α={self.alpha:.4f}, ρ={self.rho:.4f}, ν={self.nu:.4f}")
        print(f"FitL RMSE={self.rmse*100:.4f}%, Max Error={self.max_error*100:.4f}%")

        #extra measure, it just tells us how well the sabr model is fitting, based on the data qyality
        #if rmse is high sabr isn't fitting well

        return self

    def get_vol(self, strike):
        """Get SABR implied vol for a given strike."""
        return hagan_sabr_vol(strike, self.forward, self.t,
                              self.alpha, self.beta, self.rho, self.nu)

    def plot(self):
        """market v sabr volatility smile."""
        k_range = np.linspace(self.spot * 0.8, self.spot * 1.2, 100)
        sabr_vols = [self.get_vol(k) * 100 for k in k_range]

        plt.figure(figsize=(10, 5))
        plt.scatter(self.strikes, self.market_vols * 100,
                    c='blue', s=40, label='Market', zorder=5)
        plt.plot(k_range, sabr_vols, 'r-', lw=2, label='SABR')
        plt.axvline(self.spot, color='gray', ls='--', alpha=0.5, label='Spot')
        plt.xlabel('Strike')
        plt.ylabel('Implied Vol (%)')
        plt.title(f'{self.ticker} Volatility Smile ({self.expiry})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def summary(self):
        """formatted calibration results"""

        if self.alpha is None:
          return "Model not calibrated."

        return (
          f"\n{'='*50}\n"
          f"SABR Model: {self.ticker}\n"
          f"{'='*50}\n"
          f"Spot:     ${self.spot:.2f}\n"
          f"Forward:  ${self.forward:.2f}\n"
          f"Expiry:   {self.expiry} ({self.t*365:.0f} days)\n"
          f"\nCalibrated Parameters:\n"
          f"  α (alpha): {self.alpha:.6f}\n"
          f"  β (beta):  {self.beta:.4f}\n"
          f"  ρ (rho):   {self.rho:.6f}\n"
          f"  ν (nu):    {self.nu:.6f}\n"
          f"\nFit Quality:\n"
          f"  RMSE:      {self.rmse*100:.4f}%\n"
          f"  Max Error: {self.max_error*100:.4f}%\n"
          f"{'='*50}"
        )

    def get_vol_surface(self, strikes=None, maturities=None, r=0.05):
        """
        Generate volatility surface data across strikes and maturities.

        Returns:
          K_mesh, T_mesh, vol_surface (2D arrays for 3D plotting)

        """

        if self.alpha is None:
          raise ValueError("model not calibrated yet")

        if strikes is None:
          strikes = np.linspace(self.spot * 0.8, self.spot*1.2, 50)

        if maturities is None:
          maturities = np.linspace(0.1, 2.0, 40)

        K_mesh, T_mesh = np.meshgrid(strikes, maturities)

        vol_surface = np.zeros_like(K_mesh)

        for i, t in enumerate(maturities):
          f = self.spot * np.exp(r * t)
          for j, k in enumerate(strikes):
            vol_surface[i, j] = hagan_sabr_vol(
                k, f, t,
                self.alpha, self.beta, self.rho, self.nu
            )

        return K_mesh, T_mesh, vol_surface

    def plot_surface(self, strikes=None, maturities=None):
      """3d vol surf plot"""

      if self.alpha is None:
        raise ValueError("Model not calibrated. Call calibrate() first.")


      K_mesh, T_mesh, vol_surface = self.get_vol_surface(strikes, maturities)

      fig = go.Figure(data=[go.Surface(
            x=K_mesh,
            y=T_mesh,
            z=vol_surface * 100,
            colorscale='Viridis',
            colorbar=dict(title='Vol (%)')
        )])

      fig.update_layout(
            title=f'{self.ticker} SABR Volatility Surface<br>(α={self.alpha:.4f}, ρ={self.rho:.4f}, ν={self.nu:.4f})',
            scene=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Time to Maturity (years)',
                zaxis_title='Implied Volatility (%)'
            ),
            width=900,
            height=700
        )

      return fig


# model = SABR("AAPL", beta=0.5)
# model.calibrate(expiry_index=5, option_type='calls')

# print(model.summary())
# model.plot()
# fig = model.plot_surface()
# fig.show()
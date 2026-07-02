# Foul Up 3 Strategy Models

This repo contains two Streamlit apps for NBA late-game intentional-fouling strategy.

## Apps

- `simple_foul_up3_threshold`: compact dashboard for 3-6 point margins, with smoothed stable foul windows.
- `foul_up_three_model`: larger exploratory Monte Carlo dashboard with more assumptions and sensitivity views.

## Run The Simple App

```powershell
cd simple_foul_up3_threshold
pip install numpy pandas plotly streamlit pytest
streamlit run app.py
```

## Run The Full App

```powershell
cd foul_up_three_model
pip install -r requirements.txt
streamlit run app.py
```

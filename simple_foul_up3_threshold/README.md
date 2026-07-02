# Simple Foul Thresholds

Tiny Streamlit app that estimates when fouling beats defending at 3-6 point margins.
Tabs show down 3, down 4, down 5, and down 6 cases.
Each tab also mirrors the question for the team leading by that same margin.
The model sweeps 1-60 seconds with 50,000 Monte Carlo trials per time.
Inputs: trailing FT%, trailing 3P%, leading FT%, FT-miss OREB%.
Hardcoded assumptions live at the top of `model.py`.
Free throws use no clock; fouls burn 3 seconds; defended possessions burn up to 10 seconds.
Down 3+, defended trailing teams attempt threes.
Down 1 or 2, trailing teams attempt a 50% two.
The trailing team fouls the leader under 30 seconds.
Ties at the buzzer are resolved 50/50 in overtime.

Run: `pip install numpy pandas plotly streamlit pytest`
Then: `streamlit run app.py`
Tests: `python -m pytest -q`

"""Quick smoke test for NB13 plotly figure creation."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 25', 'Bottom 25'), horizontal_spacing=0.12)
fig.add_trace(go.Bar(x=[90,95,110], y=['A','B','C'], orientation='h', name='High'), row=1, col=1)
fig.add_trace(go.Bar(x=[80,85,88], y=['D','E','F'], orientation='h', name='Low'), row=1, col=2)
try:
    fig.add_vline(x=100, line_dash='dash', line_color='gray', row=1, col=1)
    print('add_vline with row/col OK')
except Exception as e:
    print('add_vline row/col FAIL:', e)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=[2015,2016,2017,2017,2016,2015],
    y=[105,108,110,95,92,90],
    fill='toself',
    fillcolor='rgba(100,100,200,0.12)',
    line=dict(color='rgba(0,0,0,0)'),
    name='band',
))
print('fill scatter OK')

fig.write_html('/tmp/test_lb.html')
print('leaderboard HTML written OK')

import plotly
from plotly import graph_objs as go
import plotly.graph_objects as go1
import pyecharts
import pandas as pd


def gen_line_plotly(img_path, df, attrs=None, names=None, consult_cols=None, **kwargs):
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    keys = list(df.columns) if names is None else names
    if attrs is None:
        attrs = [f'att{i}' for i in range(len(df))]
    elif len(attrs) != len(df):
        raise Exception('attrs length error.')
    else:
        attrs = [f'_{i}' for i in attrs]
    ####
    data = [go.Scatter(x=attrs, y=df[keys[i]], name=keys[i]) for i in range(len(keys))]
    if consult_cols:
        data += [go.Scatter(x=attrs, y=line, name='-', line={'dash': 'dot'}) for line in consult_cols]
    plotly.offline.plot({
        "data": data,
        "layout": go.Layout(title=kwargs.get('graph_name', '-'))
    },
        filename=img_path,
        auto_open=kwargs.get('auto_open', False)
    )


def draw_bar_plotly(keys, df):
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    names = df.columns
    fig = go.Figure(data=[go.Bar(x=keys, y=df[name], name=name) for name in names])
    fig.show()


def draw_bar_plotly_by_counter(c):
    _tdf = pd.DataFrame({'k': [x for x in c], 'v': [c.get(x) for x in c]})
    _tdf1 = _tdf.sort_values(by='v')
    fig = go.Figure(data=[go.Bar(x=_tdf1.k, y=_tdf1.v)])
    fig.show()


def draw_lines(xs, ys, names):
    fig = go.Figure()
    # Add traces
    for y, name in zip(ys, names):
        fig.add_trace(go.Scatter(x=xs, y=y,
                    mode='lines+markers',
                    name=name))
    fig.show()


def draw_scatters(xs, ys, names):
    fig = go.Figure()
    # Add traces
    for x, y, name in zip(xs, ys, names):
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='markers',
                    name=name))
    fig.show()

if __name__ == '__main__':
    animals = ['giraffes', 'orangutans', 'monkeys']
    df1 = {
        'a': [1, 3, 4],
        'b': [2, 2, 2],
    }
    draw_bar_plotly(animals, df1)
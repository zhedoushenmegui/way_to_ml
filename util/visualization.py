import plotly
from plotly import graph_objs as go
import plotly.graph_objects as go1
import pyecharts
import pandas as pd
import os


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
    if isinstance(ys, pd.DataFrame):
        ys = [ys[k] for k in ys.columns]
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


def tree_to_dot(tree_dict, keys=None):
    # node: 0 [label="abc"];
    # line: 0 -> 1;
    if keys is None:
        keys = ['key', 'val', 'size', 'reason']
    nodes = []
    lines = []
    index = [0]
    def _process(struct, parent_id=None, path=''):
        cnt = '\\n'.join([f'{key}:{struct.get(key, "-")}' for key in keys])
        node = f'{index[0]} [label="{cnt}"];'
        nodes.append(node)
        if parent_id is not None:
            if path:
                lines.append(f'{parent_id} -> {index[0]} [headlabel="{path}"];')
            else:
                lines.append(f'{parent_id} -> {index[0]};')
        ###
        selfindex = index[0]
        index[0] += 1
        if 'trees' in struct:
            for v in struct['trees']:
                _process(struct['trees'][v], selfindex, path=v)
    ###
    _process(tree_dict)
    nodesstr = '\n'.join(nodes)
    linesstr = '\n'.join(lines)
    outputstr = 'digraph Tree {\nnode [shape=box] ;\n %s\n%s\n}' % (nodesstr, linesstr)
    return outputstr


## 这里dot 是一个绘图命令
## open 是mac 上的打开文件命令
def draw_tree(tree, save_path, keys=None, auto_open=False):
    s = tree_to_dot(tree, keys)
    with open(f'{save_path}.dot', 'w') as f:
        f.write(s)
    os.system(f'dot -Tpng {save_path}.dot -o {save_path}.png && open {save_path}.png')


if __name__ == '__main__':
    animals = ['giraffes', 'orangutans', 'monkeys']
    df1 = {
        'a': [1, 3, 4],
        'b': [2, 2, 2],
    }
    draw_bar_plotly(animals, df1)

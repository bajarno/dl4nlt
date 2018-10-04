import torch
import plotly
import plotly.graph_objs as go
import  pickle


def create_graph(title, content, attention):
    attention = attention*255
    colors = [['#FFFFFF'] * len(content)] + [[rgb2hex(255-int(c), 255-int(c*0.5), 255) for c in l] for l in attention.tolist()]

    trace = go.Table(
        header=dict(
            values=['']+title,
            line = dict(color = 'white'),
            align = ['center', 'center']),
        cells=dict(
            values=[content]+['']*attention.size()[0], #attention.tolist(),
            fill=dict(color=colors),
            line = dict(color = 'white'),
            align = ['right', 'center'])
        )

    # pio.write_image(trace, 'plotly_test.jpeg')

    data = [trace]
    plotly.offline.plot(data, filename='plotly_test')

def rgb2hex(r,g,b):
    return '#%02x%02x%02x' % (r, g, b)

if __name__ == '__main__':
    # title = ['<s>', 'This', 'is', 'a', 'title', 'with', 'something', 'about', 'trump', '.', '</s>']
    # content = ['<s>', 'For', 'this', 'article', 'we', 'provided', 'some', 'dummy', 'content', '.', 'A', 'lot', 'of', 'politics', 'is', 'involved', '.', '</s>']
    # attention_matrix = (torch.empty(len(title), len(content)).uniform_()*255).int()

    pickle_file = open('../data/example_attn.pkl', 'rb')
    data = pickle.load(pickle_file)

    title = data['pred']
    content = data['x']
    attention_matrix = torch.from_numpy(data['weights'])
    attention_matrix = attention_matrix.transpose(0,1)

    create_graph(
        title,
        content,
        attention_matrix)
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')

def read_file(fn):
    stats = []
    with open(fn, 'r') as f:
        stats = [line.split()[2:4] for line in f]

    return stats

def identify_log_files(dirname):
    return [filename for _, _, files in os.walk(dirname) for filename in files  if filename[-3:] == 'log']

def add_to_plot(a, stats, params):
    loss, acc = zip(*stats)
    a.plot(loss, label='{enc}, hidden={hidden}, lr={lr}, tf={tf:.2f}, order={order}'.format(**params))


def parse_filename(fn):
    fields  = fn[:-4].split('_')
    params = dict()
    params['enc'] = fields[0]
    params['hidden'] = int(fields[3])
    params['order'] = int(fields[5])
    params['lr'] = float(fields[7])
    params['tf'] = float(fields[9])
    return params

def plot_all_stats(dirname):
    files = identify_log_files(dirname)
    print(files)

    f = figure()
    a = f.gca()
    for fn in files:
        params = parse_filename(fn)
        stats = read_file(dirname + '/' + fn)
        add_to_plot(a, stats, params)

    a.legend()
    show()

if __name__ == "__main__":
    dirname = '../logs'
    plot_all_stats(dirname)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import os

alias = {}
styles = {}
colors = {}


def parse_index_name(iname):
    # parse '*.map-DFS-3-c4-inv' to itype="inv", d=4, hlevel=3
    # or '*.map-DFS-3-c4' to itype="vanilla", d=4, hlevel=3
    ps = iname.split('-')
    itype = "forward"
    if ps[-1] == "inv":
        itype = "backward"
        ps.pop()
    r = int(ps[-1][1:]) if ps[-1][0] == 'c' else 0
    ps.pop()
    ps.pop()
    ps.pop()
    mname = '-'.join(ps)[:-4]
    return dict(map=mname,itype=itype,r=r)

def load_size(fname, filter=True):
    with open(fname, 'r') as f:
        raw = f.readlines()
    rows = []
    game_maps = os.listdir("../maps/gppc")
    header = ["map", "itype", "r", "size"]
    for line in raw:
        size, path = line.split()
        data = parse_index_name(path)
        mapname = data['map'].split('/')[-1]
        if not filter or (mapname + ".map") in game_maps: 
            rows.append([mapname, data['itype'], data['r'], size])
    df = pd.DataFrame.from_records(rows, columns=header)
    df = df[header].apply(pd.to_numeric, errors="ignore")
    return df

def load_data(fname):
    with open(fname, "r") as f:
        # ignore `vertices: ...` at head and `Total: ..` at tail
        raws = f.readlines()
    header = [i.strip() for i in raws[0].split(',')]
    lines = [[j.strip() for j in i.split(',')] for i in raws[1:]]
    t = pd.DataFrame.from_records(lines, columns=header)
    res = t[header].apply(pd.to_numeric, errors='ignore')
    return res

def load_files(paths):
    frames = []
    for i in paths:
        print(i)
        frames.append(load_data(i))
    res = pd.concat(frames)
    return res

def gen_xy(df=None, colx='', coly='', ignore=True, limit=20):
    tg = df.groupby(colx)
    x = []
    y = []
    for k, v in tg[coly].apply(lambda _: np.average(_)).items():
        if ignore and tg.size()[int(k)] < limit:
            continue
        x.append(int(k))
        y.append(v)

    return x, y

def gen_xy_median(df=None, colx='', coly='', ignore=True, limit=20):
    tg = df.groupby(colx)
    x = []
    y = []
    for k, v in tg[coly].apply(lambda _: np.median(_)).items():
        if ignore and tg.size()[int(k)] < limit:
            continue
        x.append(int(k))
        y.append(v)

    return x, y


def gen_xy_percentile(df=None, colx='', coly='', ignore=True, limit=20):
    tg = df.groupby(colx)
    x = []
    y = []
    for k, v in tg[coly].apply(lambda _: np.percentile(_,95)).items():
        if ignore and tg.size()[int(k)] < limit:
            continue
        x.append(int(k))
        y.append(v)

    return x, y


def plot_graph_percent(title ='', xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None, yticks=None,
               loc='out',length=0):

    fig, ax = plt.subplots(figsize=(7,5))

    # ax.xaxis.set_ticks(np.arange(0, length+length/4, length/4))
    ax.xaxis.set_major_locator(MultipleLocator(length/4))
    ax.xaxis.set_minor_locator(MultipleLocator(length/10))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=length, decimals=None, symbol='', is_latex=False))
    ax.set_xlabel(xlabel, weight ='bold',size ='16')
    ax.set_ylabel(ylabel,weight ='bold',size ='16')

    ax.set_yscale(yscale)
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 14}
    plt.rc('font',**font)
    plt.rc("text", usetex=False)
    plt.title(title,fontweight="bold",size ='14')
    if xscale is not None:
        ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        print(xticks)
        plt.xticks(xticks[0], xticks[1])
    if yticks is not None:
        plt.yticks(yticks[0], yticks[1])



    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        if styles.get(labels[i]) is not None:
            if colors.get(labels[i]) is not None:
                plt.plot(x, y, color =colors.get(labels[i]), linestyle =styles[labels[i]], label=labels[i],linewidth=2)
            else:
                plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)

        else:
            # ax.scatter(x, y)

            ax.plot(x, y, label=labels[i])
    ax.legend(labels)
    if loc == 'in':
        ax.legend(loc='upper left' , ncol=2,fancybox=True, framealpha=0, prop={'size': 14})

    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, prop={'size': 14 ,'weight':'bold'})
    plt.grid(True,which='both',linestyle='-', linewidth=0.01,color ='gainsboro')
    if saveto is not None:
        fig.savefig(saveto, bbox_inches='tight')


def plot_ttf(title ='', xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None, yticks=None,
               loc='out',length=0, padding= 0, separator = 1,corrdinate= None):

    fig, ax = plt.subplots(figsize=(4,2))

    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.set_xlabel(xlabel, weight ='bold',size ='12',labelpad = padding)
    ax.set_ylabel(ylabel,weight ='bold',size ='12',labelpad = padding)

    ax.set_yscale(yscale)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 12}
    plt.rc('font',**font)
    plt.rc("text", usetex=False)
    # plt.title(title,fontweight="bold",size ='22',pad=30)
    plt.title(title,fontweight="bold",size ='14')
    if xscale is not None:
        ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        print(xticks)
        plt.xticks(xticks[0], xticks[1],padding =20)

    if yticks is not None:
        plt.yticks(yticks[0], yticks[1],padding= 20)

    # plt.tick_params(axis='both', which='major', labelsize=16)
    # ax.tick_params(axis='both', which='major', pad=15, labelsize=12)
    # plt.xticks(np.arange(0, 25, 4))
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        ax.scatter(x, y)
        ax.plot(x, y)
        plt.plot(x, y, 'D')
        index = 0
        for xy in zip(x, y):
            ax.annotate('(%s,%s)' % xy, xy=xy, xytext = corrdinate[index])
            index += 1


    plt.grid(True,which='both',linestyle='-', linewidth=0.01,color ='gainsboro')
    if saveto is not None:
        fig.savefig(saveto, bbox_inches='tight')
        
def plot_graph_k(title ='', xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None, yticks=None,
               loc='out',length_start=0,length=0, padding= 0, separator = 1, position=None, image_name='',image_extent=[]):
    import matplotlib.image as mpimg

    
   
    fig, ax = plt.subplots(figsize=(5,5))

    if image_name != '':
        im = mpimg.imread(image_name)
        ax.imshow(im, extent=image_extent, aspect='auto',zorder=3)

        
    ax.xaxis.set_ticks(np.arange(length_start, length_start+length+length/5, length/5))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(length))
#     ax.xaxis.set_major_locator(MultipleLocator(25))
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel(xlabel, weight ='bold',size ='12',labelpad = padding)
    ax.set_ylabel(ylabel,weight ='bold',size ='12',labelpad = padding)
    ax.set_yscale(yscale)
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 12}
    plt.rc('font',**font)
    plt.rc("text", usetex=False)
    # plt.title(title,fontweight="bold",size ='22',pad=30)
    plt.title(title,fontweight="bold",size ='14')
    if xscale is not None:
      ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
      print(xticks)
      plt.xticks(xticks[0], xticks[1],padding =20)

    if yticks is not None:
      plt.yticks(yticks[0], yticks[1],padding= 20)

    
    # plt.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', pad=15, labelsize=12)
    # plt.xticks(np.arange(0, 25, 4))
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        # if styles.get(labels[i]) is not None:
        #     plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)
        # else:
        #     ax.plot(x, y, label=labels[i])
#         plt.gca().set_yticklabels(['{:,.0%}'.format(i) for i in y])
        
        if styles.get(labels[i]) is not None:
            if colors.get(labels[i]) is not None:
                plt.plot(x, y, color =colors.get(labels[i]), linestyle =styles[labels[i]], label=labels[i],linewidth=2)
            else:
                plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)

        else:
            # ax.scatter(x, y)

            ax.plot(x, y, label=labels[i])
    ax.legend(labels)
    if loc == 'in':
      # ax.legend(loc='upper left',fancybox=True, framealpha=0, prop={'size': 12})
        if position == 'up':
            ax.legend(loc='upper left',fancybox=True, framealpha=0,ncol=2, prop={'size': 12})
        else:
            ax.legend(loc='lower left',fancybox=True, framealpha=0,ncol=2, prop={'size': 12})
            
    else:
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, prop={'size': 12 ,'weight':'bold'})
    plt.grid(True,which='both',linestyle='-', linewidth=0.01,color ='gainsboro',zorder=2)
    ax.set_axisbelow(True)
    plt.rc('axes', axisbelow=True)

    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2f}'.format(x/1000) + 'k' for x in current_values])
#     ax.set_yticks(tick_loc)
#     ticks = ['{:,.2f}'.format(x/1000) + 'k'for x in ticks_loc]
#     ticks[0] = -1
#     ax.set_yticklabels(ticks)
    
#     current_values = plt.gca().get_yticks()
#     ax.set_yticklabels(['{:,.2f}'.format(x/1000) + 'k' for x in current_values])
#     print(current_values)
    # using format string '{:.0f}' here but you can choose others

    if saveto is not None:
        fig.savefig(saveto, bbox_inches='tight')
        
        
def plot_graph_percentage(title ='', xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None, yticks=None,
               loc='out',length_start=0,length=0, padding= 0, separator = 1, position=None, image_name='',image_extent=[]):
    import matplotlib.image as mpimg

    
   
    fig, ax = plt.subplots(figsize=(5,5))

    if image_name != '':
        im = mpimg.imread(image_name)
        ax.imshow(im, extent=image_extent, aspect='auto',zorder=3)

        
    ax.xaxis.set_ticks(np.arange(length_start, length_start+length+length/5, length/5))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(length))
#     ax.xaxis.set_major_locator(MultipleLocator(25))
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel(xlabel, weight ='bold',size ='12',labelpad = padding)
    ax.set_ylabel(ylabel,weight ='bold',size ='12',labelpad = padding)

    ax.set_yscale(yscale)
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 12}
    plt.rc('font',**font)
    plt.rc("text", usetex=False)
    # plt.title(title,fontweight="bold",size ='22',pad=30)
    plt.title(title,fontweight="bold",size ='14')
    if xscale is not None:
      ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
      print(xticks)
      plt.xticks(xticks[0], xticks[1],padding =20)

    if yticks is not None:
      plt.yticks(yticks[0], yticks[1],padding= 20)

    
    # plt.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', pad=15, labelsize=12)
    # plt.xticks(np.arange(0, 25, 4))
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        # if styles.get(labels[i]) is not None:
        #     plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)
        # else:
        #     ax.plot(x, y, label=labels[i])
#         plt.gca().set_yticklabels(['{:,.0%}'.format(i) for i in y])
        if styles.get(labels[i]) is not None:
            if colors.get(labels[i]) is not None:
                plt.plot(x, y, color =colors.get(labels[i]), linestyle =styles[labels[i]], label=labels[i],linewidth=2)
            else:
                plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)

        else:
            # ax.scatter(x, y)

            ax.plot(x, y, label=labels[i])
    ax.legend(labels)
    if loc == 'in':
      # ax.legend(loc='upper left',fancybox=True, framealpha=0, prop={'size': 12})
        if position == 'up':
            ax.legend(loc='upper left',fancybox=True, framealpha=0,ncol=2, prop={'size': 12})
        else:
            ax.legend(loc='lower left',fancybox=True, framealpha=0,ncol=2, prop={'size': 12})
            
    else:
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, prop={'size': 12 ,'weight':'bold'})
    plt.grid(True,which='both',linestyle='-', linewidth=0.01,color ='gainsboro',zorder=2)
    ax.set_axisbelow(True)
    plt.rc('axes', axisbelow=True)
    

#     ax.margins(y=0)
#     current_values = plt.gca().get_yticks()
#     # using format string '{:.0f}' here but you can choose others
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0%}'.format(x) for x in current_values])
    if saveto is not None:
        fig.savefig(saveto, bbox_inches='tight')
        
def plot_graph(title ='', xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None, yticks=None,
               loc='out',length_start=0,length=0, padding= 0, separator = 1, position=None, image_name='',image_extent=[]):
    import matplotlib.image as mpimg

    
   
    fig, ax = plt.subplots(figsize=(5,5))

    if image_name != '':
        im = mpimg.imread(image_name)
        ax.imshow(im, extent=image_extent, aspect='auto',zorder=3)

        
    ax.xaxis.set_ticks(np.arange(length_start, length_start+length+length/5, length/5))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(length))
#     ax.xaxis.set_major_locator(MultipleLocator(25))
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel(xlabel, weight ='bold',size ='12',labelpad = padding)
    ax.set_ylabel(ylabel,weight ='bold',size ='12',labelpad = padding)

    ax.set_yscale(yscale)
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 12}
    plt.rc('font',**font)
    plt.rc("text", usetex=False)
    # plt.title(title,fontweight="bold",size ='22',pad=30)
    plt.title(title,fontweight="bold",size ='14')
    if xscale is not None:
      ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
      print(xticks)
      plt.xticks(xticks[0], xticks[1],padding =20)

    if yticks is not None:
      plt.yticks(yticks[0], yticks[1],padding= 20)

    # plt.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', pad=15, labelsize=12)
    # plt.xticks(np.arange(0, 25, 4))
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        # if styles.get(labels[i]) is not None:
        #     plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)
        # else:
        #     ax.plot(x, y, label=labels[i])

        if styles.get(labels[i]) is not None:
            if colors.get(labels[i]) is not None:
                plt.plot(x, y, color =colors.get(labels[i]), linestyle =styles[labels[i]], label=labels[i],linewidth=2)
            else:
                plt.plot(x, y, linestyle =styles[labels[i]], label=labels[i],linewidth=2)

        else:
            # ax.scatter(x, y)

            ax.plot(x, y, label=labels[i])
    ax.legend(labels)
    if loc == 'in':
      # ax.legend(loc='upper left',fancybox=True, framealpha=0, prop={'size': 12})
        if position == 'up':
            ax.legend(loc='upper left',fancybox=True, framealpha=0,ncol=2, prop={'size': 12})
        else:
            ax.legend(loc='lower left',fancybox=True, framealpha=0,ncol=2, prop={'size': 12})
            
    else:
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, prop={'size': 12 ,'weight':'bold'})
    plt.grid(True,which='both',linestyle='-', linewidth=0.01,color ='gainsboro',zorder=2)
    ax.set_axisbelow(True)
    plt.rc('axes', axisbelow=True)
    if saveto is not None:
        fig.savefig(saveto, bbox_inches='tight')



def plot_point_graph(title ='', xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None, yticks=None,
               loc='out',length=0, padding= 0, separator = 1):

    fig, ax = plt.subplots(figsize=(15,5))

    # ax.xaxis.set_ticks(np.arange(0, length+length/4, length/4))
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(length))

    ax.set_xlabel(xlabel, weight ='bold',size ='12',labelpad = padding)
    ax.set_ylabel(ylabel,weight ='bold',size ='12',labelpad = padding)

    ax.set_yscale(yscale)
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 12}
    plt.rc('font',**font)
    plt.rc("text", usetex=False)
    plt.title(title,fontweight="bold",size ='22',pad=30)
    if xscale is not None:
        ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        print(xticks)
        plt.xticks(xticks[0], xticks[1],padding =20)

    if yticks is not None:
        plt.yticks(yticks[0], yticks[1],padding= 20)

    # plt.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', pad=15, labelsize=12)
    plt.xticks(np.arange(0, length, separator))
    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        if styles.get(labels[i]) is not None:
            plt.plot(x, y, '.')
        else:
            ax.plot(x, y, '.', label=labels[i])
    ax.legend(labels)
    if loc == 'in':
        ax.legend(loc='upper left',fancybox=True, framealpha=0, prop={'size': 12})
        # ax.legend(loc='upper left',fancybox=True, framealpha=0,ncol=4, prop={'size': 12})
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, prop={'size': 12 ,'weight':'bold'})
    plt.grid(True,which='both',linestyle='-', linewidth=0.01,color ='gainsboro')
    if saveto is not None:
        fig.savefig(saveto, bbox_inches='tight')


def plot_graph_2(xlabel='', ylabel='', xs=[[]], ys=[[]], labels=[], color=None,
               yscale='log', xscale=None, ylim=None, xlim=None, saveto=None, xticks=None,
               loc='out'):

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 22}
    plt.rc('font',**font)
    plt.rc("text", usetex=True)
    if xscale is not None:
        ax.set_xscale(xscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        print(xticks)
        plt.xticks(xticks[0], xticks[1])

    n = len(xs)
    for i in range(n):
        x = xs[i]
        y = ys[i]
        ax.plot(x, y, label=labels[i])
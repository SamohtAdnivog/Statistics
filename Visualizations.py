import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from yellowbrick.features import PCA
import re

sns.set_theme()


def heatmap(X, y, stats, A_Name, B_Name, filename, features):
    if features > X.shape[0]:
        features = X.shape[0]
    stats_top = stats.head(features).index
    Xt = X.T
    X_top = pd.DataFrame()
    for ind in stats_top:
        a = Xt.loc[ind]
        X_top = pd.concat([X_top, a], axis=1, join='outer')
    labels_list = y
    colours = []
    col1 = 'forestgreen'
    col2 = 'firebrick'
    for label in labels_list:
        if label == 0:
            colours.append(col1)
        elif label == 1:
            colours.append(col2)

    fig = sns.clustermap(X_top.T, linewidths=0.5,
                         cmap="vlag",
                         dendrogram_ratio=0.07,
                         colors_ratio=0.025,
                         cbar_pos=(1.01, 0.65, 0.018, 0.25),
                         # cbar_kws= {'label': None},
                         method='ward',
                         metric='euclidean',
                         col_colors=colours,
                         xticklabels=1,
                         figsize=(20, 20)
                         )
    ax = fig.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Adding a Legend for the groups
    lut = {A_Name: col1, B_Name: col2}
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut,
               bbox_to_anchor=(1.02, 1),
               bbox_transform=plt.gcf().transFigure,
               loc='upper right',
               fontsize=13,
               framealpha=0.6,
               title='Group',
               title_fontsize=13,
               facecolor='ghostwhite'
               )
    plt.ylabel("normalized Abundance", fontsize=10, verticalalignment='top')
    fig.savefig(filename + '_heatmap.tiff', dpi=300, orientation="letter", bbox_inches="tight")
    plt.close()


def PCA_Anal(X, y, A_Name, B_Name, classes, filename):
    sns.set_theme(style="whitegrid")
    visualizer = PCA(scale=True, proj_features=False, projection=2, classes=classes)
    x = visualizer.fit_transform(X, y)
    col1 = 'dodgerblue'
    col2 = 'salmon'

    g = sns.relplot(x=x[:, 0],
                    y=x[:, 1],
                    hue=y,
                    legend=False,
                    palette=[col1, col2],
                    )
    g.set_axis_labels("Principal Component 1", "Principal Component 2")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    # g.despine(left=True, bottom=True)

    # Adding a Legend for the groups
    lut = {A_Name: col1, B_Name: col2}
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut,
               bbox_to_anchor=(1.1, 1),
               bbox_transform=plt.gcf().transFigure,
               loc='upper right',
               fontsize=12,
               framealpha=0.6,
               title=False,
               title_fontsize=13,
               facecolor='ghostwhite'
               )

    plt.savefig(filename + '_PCA.tiff', dpi=300, orientation="letter", bbox_inches="tight")
    plt.close()


def heatmap_ml(method):
    df = pd.read_csv('machine_learning_results.csv', index_col='Comparison / CV Scores')
    df.drop(columns='Unnamed: 0', inplace=True)
    ax = sns.heatmap(df,
                     annot=True,
                     linewidths=0.8,
                     cmap='YlGnBu',
                     xticklabels=['BNC', 'kNNC', 'RFC', 'lSVMC'],
                     )
    plt.yticks(rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.savefig("ML_heat" + method + ".tiff", dpi=300, orientation="letter", bbox_inches="tight")
    plt.close()


def volcano_ann(file, filename):
    df = pd.read_csv(file)
    # Getting Species
    species = df["Unnamed: 0"]
    species_short = []

    for item in species:
        species_short.append(item[:-5])
    species = []
    for item in species_short:
        a = (re.sub(r'\d+', '', item)).strip()
        species.append(a)
    series = pd.Series(species, name="group")
    df = pd.concat([df, series], axis=1)
    df.set_index("Unnamed: 0", inplace=True)
    x = "log2FoldChange"
    y = "logpvalue"

    f, ax = plt.subplots(figsize=(8, 8.5))
    # Adding borders
    pt = np.arange(np.min(df['log2FoldChange']) - 0.5, np.max(df['log2FoldChange']) + 0.5, 0.01)
    ps = []
    for item in pt:
        ps.append(-np.log10(0.05))
    fs = np.arange(0, np.max(df['logpvalue']) + 0.5, 0.01)
    ftr = []
    for item in fs:
        ftr.append(1)
    ftl = []
    for item in fs:
        ftl.append(-1)
    # Style
    sns.despine(f, left=True, bottom=True)
    ax.set_xlabel(r"log$_2$(FoldChange)", fontsize=15, fontweight='bold')
    ax.set_ylabel("-log$_{10}$(p-value)", fontsize=15, fontweight='bold')
    # ax.set_title("Volcano plot", fontsize=20)
    sns.scatterplot(data=df,
                    x=x,
                    y=y,
                    sizes=(1, 13),
                    hue="group",
                    linewidth=1,
                    alpha=0.9,
                    ax=ax,
                    legend='auto',
                    edgecolor='black',
                    )
    ax.plot(pt, ps, 'k--', alpha=0.7, linewidth=1)
    ax.plot(ftr, fs, 'k--', alpha=0.7, linewidth=1)
    ax.plot(ftl, fs, 'k--', alpha=0.7, linewidth=1)
    plt.title(filename)

    ax.legend(
        bbox_to_anchor=(1.02 ,1),
        borderaxespad=0,
        title="Lipids", )
    plt.savefig(filename + "_Volcano.tiff", dpi=300, orientation="letter", bbox_inches="tight")
import pandas as pd
import numpy as np  
import seaborn as sns
import pylab   as plt

def heatmap(input_df, cluster_level):
    
    heatmap_df=input_df.groupby(cluster_level).mean()
        
    for c in heatmap_df.columns:
        heatmap_df[c]=heatmap_df[c]/input_df[c].mean()-1
    heatmap_df=heatmap_df*100
    
    plt.figure(figsize=(14,float(len(input_df[cluster_level].unique()))/2.5))
    cmap=sns.diverging_palette(238, 12, l=60, s=100, as_cmap=True)
    sns.heatmap(heatmap_df, cbar_kws={'format': '%.0f%%', 'aspect':2.5}, vmax=500, cmap=cmap, center=0)
    plt.ylabel('Clusters')
    plt.savefig("heatmap_{}.pdf".format(cluster_level), format="pdf")
    plt.show()
    return


df = pd.read_csv('full_simple_elixhauser_comorbidity.csv')
heatmap(df, 'cluster')


import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def main():
    plt.style.use('seaborn-white')
        # matplotlib.rc("font", family="Ubuntu Sans")

    dataframe = pandas.read_excel('ensemble5.xlsx', sheet_name='table')
    # Remove row with labels
    performance_values = dataframe.values[1:, :].astype(np.float32)
    stages = ['W', 'DS', 'N3', 'REM']
    precision = [0,3,6,9]
    recall = [1,4,7,10]
    f1 = [2,5,8,11]

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(12,6))
    bp1 = ax1.boxplot(performance_values[:, precision], patch_artist=True)
    bp2 = ax2.boxplot(performance_values[:, recall], patch_artist=True)
    bp3 = ax3.boxplot(performance_values[:, f1], patch_artist=True)

    ax1.set_title("Precision",fontsize=18)
    ax2.set_title("Recall",fontsize=18)
    ax3.set_title("F-1 score",fontsize=18)

    for ax in f.axes:
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(False) 
        ax.get_xaxis().tick_bottom()    
        ax.get_yaxis().tick_left()   
        ax.set_xticklabels(stages) 
        ax.tick_params(labelsize=14)

    for ax in f.axes[1:]:
        ax.set_yticklabels([])
    
    plt.subplots_adjust(wspace=0.1)
    
    colors = ['gray', 'blue', 'green']
    bplots = [bp1, bp2, bp3]
    for (bp, color) in zip(bplots, colors):
        for key in ['boxes', 'whiskers', 'caps', 'medians']:
            set_color(bp[key], color)

    for patch in bp1['medians']:
        patch.set_color('lightgray')    

    for patch in bp2['medians']:
        patch.set_color('lightblue')

    for patch in bp3['medians']:
        patch.set_color('lightgreen')

    for flier in bp1['fliers']:
        flier.set(marker='x', markeredgecolor='gray', alpha=0.5)

    for flier in bp2['fliers']:
        flier.set(marker='x', markeredgecolor='blue', alpha=0.5)

    for flier in bp3['fliers']:
        flier.set(marker='x', markeredgecolor='green', alpha=0.5)

    for ax in f.axes:
        for y in range(0, 11):    
            ax.plot(range(0, 6), [y * 0.1] * len(range(0, 6)), "--", lw=0.5, color="black", alpha=0.3)  
    
    # plt.tick_params(axis="both", which="both", bottom="off", top="off",    
    #             labelbottom="on", left="off", right="off", labelleft="on")  
    # plt.figure()
    # p = plt.boxplot(performance_values[:, recall])
    # plots.append(p)

    # plt.figure()
    # p = plt.boxplot(performance_values[:, f1])
    # plots.append(p)
    plt.show()

def set_color(objects, color):
    for obj in objects:
        obj.set(color=color, linewidth=2)

if __name__ == '__main__':
    main()
import torch
import matplotlib.pyplot as plt
import csv

def plot_acc(file,  
             x_label, 
             x_scale_factor = 1.0,
             end_index=None,
             x_lims = None, 
             y_lims = None, 
             x_ticks = None, 
             y_ticks = None, 
             difference = False, 
             log = False
             ):
    '''
    for plotting accuracy performances for 1-5 layer system
    with uncertainy

    for the formatting expected for input see xy_misalignment 
    '''
    
    colors = [
    "#6FA8DC",  # stronger blue
    "#E06666",  # muted red
    "#93C47D",  # green
    "#F6B26B",  # orange
    "#B4A7D6",  # purple
    "#FFD966"   # yellow
]   

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    header = x_scale_factor * torch.tensor( 
                                            [float(x) for x in data[0][1:]]
                                            )
    
    rows = torch.tensor(
    [[float(x) for x in row[1:]] for row in data[1:]],
    )


    if difference is True:
        # assume the aligned system is the first one
        rows = rows - rows[:,0].unsqueeze(1)
    
    for i,item in enumerate(colors):

        plt.errorbar( header[:end_index], 
                 rows[ 2*i , :end_index ], 
                 yerr= rows[ 2*i +1 , :end_index],
                 linewidth=1, 
                 label=f'{i+1} Layers', 
                 color = item, marker='o',
                 capsize=5
                 )

    plt.xlabel(x_label)

    if difference == True:
        plt.ylabel('Difference in Accuracy from ideal (%)')
    else:
        plt.ylabel('Raw Accuracy (%)')

    if x_lims != None: 
        plt.xlim(x_lims)
    if y_lims != None:
        plt.ylim(y_lims)
    if x_ticks != None:
        plt.xticks(x_ticks)
    if y_ticks != None:
        plt.yticks(y_ticks)

    if log == True:
        plt.xscale('log')

    plt.legend()
    plt.show()

def plot_abs_and_diff(file,  
             x_label, 
             x_scale_factor = 1.0,
             end_index=None,
             x_lims = None, 
             y_lims = None, 
             x_ticks = None, 
             y_ticks = None, 
             log = False
             ):
    
    plot_acc(file,
             x_label,
             x_scale_factor,
              end_index,
             x_lims, 
             y_lims, 
             x_ticks, 
             y_ticks, 
              False,
             log
             )
    
    plot_acc(file,
             x_label,
             x_scale_factor,
              end_index,
             x_lims, 
             y_lims, 
             x_ticks, 
             y_ticks, 
             True,
             log
             )
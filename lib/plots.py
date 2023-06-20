import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import scipy.stats as stats

def blandaltman_plot(method1_measurements,method2_measurements,location,nr_of_speeds ):
    # Compute the differences and averages
    ba_palette = sns.color_palette("mako", n_colors=3) 
    hex_colors = [mcolors.rgb2hex(color) for color in ba_palette]

    method1_measurements = np.array(method1_measurements)
    method2_measurements = np.array(method2_measurements)

    differences = method1_measurements - method2_measurements
    averages = (method1_measurements + method2_measurements) / 2

    # Calculate mean difference and standard deviation
    median_difference = np.median(differences)
    mean_difference = median_difference #Because many changes
    std_deviation = np.std(differences)

    # Calculate standard error of the mean difference
    standard_error = std_deviation / np.sqrt(len(differences))

    # Calculate the limits of agreement
    lower_limit = median_difference - (1.96 * std_deviation)
    upper_limit = median_difference + (1.96 * std_deviation)

    # Calculate confidence intervals
    ci_mean_difference = stats.t.interval(0.95, len(differences)-1, loc=median_difference, scale=stats.sem(differences))
    ci_lower_limit = ci_mean_difference[0]
    ci_upper_limit = ci_mean_difference[1]

    print("CI lower limit ",ci_lower_limit)
    print("CI upper limit ",ci_upper_limit)
    print('median algo ',median_difference)
    
    mean_difference_measurement = 0.067916818
    ci_mean_difference_measurements = stats.t.interval(0.95, len(differences)-1, loc=mean_difference_measurement, scale=stats.sem(differences))
    print("CI mean limits manual measurement ",ci_mean_difference_measurements)


    # Plot the Bland-Altman plot
    #plt.scatter(averages, differences, color=hex_colors[2], s=60, alpha=0.6)
    plt.axhline(y=median_difference, color=hex_colors[1], linestyle='--')
    plt.axhline(y=lower_limit, color=hex_colors[0], linestyle='--')
    plt.axhline(y=upper_limit, color=hex_colors[0], linestyle='--')
    l,r=0,0
    for i in range(len(averages)):
        color_index = (i // nr_of_speeds) % 2  # Alternates between 0 and 1 every 5 values
        if color_index == 0: 
            color = hex_colors[2]
            r +=1 #absolutely stupid but it does work.
        else:
            color = hex_colors[1]
            l +=1

        if l == 1:
            plt.scatter(averages[i], differences[i], color=color, s=60, alpha=0.6, label="Left")
        if r==1:
            plt.scatter(averages[i], differences[i], color=color, s=60, alpha=0.6, label="Right")

        else:
            plt.scatter(averages[i], differences[i], color=color, s=60, alpha=0.6)
        
    

    # Get the x-axis limits from the figure
    x_min, x_max = plt.xlim()

    # Plot the confidence interval as a shaded area
    plt.fill_between([x_min, x_max], ci_lower_limit, ci_upper_limit, color=hex_colors[1], alpha=0.3)
    plt.xlim(x_min, x_max)

    # Add labels and title
    plt.xlabel('Average of Measurements [mm]')
    plt.ylabel('Difference between Measurements [mm]')
    plt.title('Bland-Altman Plot', fontsize=18, fontweight='bold')

    plt.legend(title='Jet Direction')

    plt.text(x_max, mean_difference-0.05, f'{mean_difference:.2f}', ha='right', va='center', color=hex_colors[1])
    plt.text(x_max, mean_difference+0.05, 'Median', ha='right', va='center', color=hex_colors[1])

    plt.text(x_max, lower_limit+0.05, '+1.96SD', ha='right', va='center', color=hex_colors[0])
    plt.text(x_max, lower_limit-0.05, f'{lower_limit:.2f}', ha='right', va='center', color=hex_colors[0])

    plt.text(x_max, upper_limit+0.05, '-1.96SD', ha='right', va='center', color=hex_colors[0])
    plt.text(x_max, upper_limit-0.05, f'{upper_limit:.2f}', ha='right', va='center', color=hex_colors[0])


    plt.savefig(location, dpi=400)
    # Show the plot
    plt.show()


def blandaltman_plot_percentage(method1_measurements,method2_measurements,location,nr_of_speeds ):
    # Compute the differences and averages
    ba_palette = sns.color_palette("mako", n_colors=3) 
    hex_colors = [mcolors.rgb2hex(color) for color in ba_palette]

    method1_measurements = np.array(method1_measurements)
    method2_measurements = np.array(method2_measurements)

    differences = method1_measurements - method2_measurements
    
    averages = (method1_measurements + method2_measurements) / 2

    differences = (differences / averages) * 100
    # Calculate mean difference and standard deviation
    median_difference = np.median(differences)
    std_deviation = np.std(differences)



    # Calculate standard error of the mean difference
    standard_error = std_deviation / np.sqrt(len(differences))

    # Calculate the limits of agreement
    lower_limit = median_difference - (1.96 * std_deviation)
    upper_limit = median_difference + (1.96 * std_deviation)

    # Calculate confidence intervals
    ci_mean_difference = stats.t.interval(0.95, len(differences)-1, loc=median_difference, scale=stats.sem(differences))
    ci_lower_limit = ci_mean_difference[0]
    ci_upper_limit = ci_mean_difference[1]


    print("CI lower limit ",ci_lower_limit)
    print("CI upper limit ",ci_upper_limit)
    print('median algo ',median_difference)

    mean_difference_measurement = 0.067916818
    ci_mean_difference_measurements = stats.t.interval(0.95, len(differences)-1, loc=mean_difference_measurement, scale=stats.sem(differences))
    print("CI mean limits manual measurement ",ci_mean_difference_measurements)


    # Plot the Bland-Altman plot
    #plt.scatter(averages, differences, color=hex_colors[2], s=60, alpha=0.6)
    plt.axhline(y=median_difference, color=hex_colors[1], linestyle='--')
    plt.axhline(y=lower_limit, color=hex_colors[0], linestyle='--')
    plt.axhline(y=upper_limit, color=hex_colors[0], linestyle='--')
    l,r=0,0
    for i in range(len(averages)):
        color_index = (i // nr_of_speeds) % 2  # Alternates between 0 and 1 every 5 values
        if color_index == 0: 
            color = hex_colors[2]
            r +=1 #absolutely stupid but it does work.
        else:
            color = hex_colors[1]
            l +=1

        if l == 1:
            plt.scatter(averages[i], differences[i], color=color, s=60, alpha=0.6, label="Left")
        if r==1:
            plt.scatter(averages[i], differences[i], color=color, s=60, alpha=0.6, label="Right")

        else:
            plt.scatter(averages[i], differences[i], color=color, s=60, alpha=0.6)

    # Get the x-axis limits from the figure
    x_min, x_max = plt.xlim()

    # Plot the confidence interval as a shaded area
    plt.fill_between([x_min, x_max], ci_lower_limit, ci_upper_limit, color=hex_colors[1], alpha=0.3)
    plt.xlim(x_min, x_max)

    # Add labels and title
    plt.xlabel('Average of Measurements [mm]')
    plt.ylabel('Difference between Measurements [%]')
    plt.title('Bland-Altman Plot', fontsize=18, fontweight='bold')

    plt.legend(title='Jet Direction')

    y_position = 8
    color = (0,0,0)
    margin = -0.05
    plt.text(x_max+margin, median_difference-y_position, f'{median_difference:.2f}', ha='right', va='center', color=color)
    plt.text(x_max+margin, median_difference+y_position, 'Median', ha='right', va='center', color=color)

    plt.text(x_max+margin, lower_limit+y_position, '+1.96SD', ha='right', va='center', color=color)
    plt.text(x_max+margin, lower_limit-y_position, f'{lower_limit:.2f}', ha='right', va='center', color=color)

    plt.text(x_max+margin, upper_limit+y_position, '-1.96SD', ha='right', va='center', color=color)
    plt.text(x_max+margin, upper_limit-y_position, f'{upper_limit:.2f}', ha='right', va='center', color=color)

    plt.savefig(location, dpi=400)
    # Show the plot
    plt.show()



def heatmap(df,palette,path,title,label):
        # Create a heatmap of the differences
    plt.figure(figsize=(6, 4))
    value = df.iloc[0, 0]
    if isinstance(value, float):
        value_type = ".2f"
    elif isinstance(value, int):
        value_type = "d"
    else:
        value_type  = ".2f"
        
    ax = sns.heatmap(df.abs(), annot=True, fmt=value_type, cmap=palette)
    plt.title(title,fontsize=18, fontweight='bold',pad=10)
    plt.xlabel('Speed [mm/min]')
    plt.ylabel('Video Index')
    y_labels = []
    for run in range(1, 5):
        y_labels.append(f"Run {run} R")
        y_labels.append(f"Run {run} L")
    plt.yticks(np.arange(len(y_labels)), y_labels,rotation=0)  
    ax.set_yticks(ax.get_yticks() + 0.5)
    plt.yticks() 
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(label)
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.show()

def boxplot(df,palette,path,title,xlabel,ylabel):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df.abs(),palette=palette)
    plt.title(title,fontsize=18, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(path, dpi=400)
    plt.show()
    
    
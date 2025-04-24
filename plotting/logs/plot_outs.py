import re
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def aggregate_results(all_results):
    """
    Aggregate results from multiple runs.
    
    Args:
        all_results: Dictionary mapping run_id -> list of parsed results
        
    Returns:
        Dictionary mapping percent -> aggregated metrics
    """
    # Collect all unique percentages across all runs
    percents = set()
    for run_results in all_results.values():
        for result in run_results:
            percents.add(result['percent'])
    
    aggregated_data = {}
    
    for percent in sorted(percents):
        # Get all results for this percentage across runs
        percent_results = []
        for run_id, run_results in all_results.items():
            for result in run_results:
                if result['percent'] == percent:
                    percent_results.append(result)
        
        if not percent_results:
            continue
            
        # Find max number of epochs across all runs for this percent
        max_epochs = max(len(r['epochs']) for r in percent_results)
        
        # Prepare lists to hold data from all runs
        all_train_losses = []
        all_val_losses = []
        all_val_accs = []
        all_rel_times = []
        all_test_accs = []
        all_total_times = []
        
        # Process each run's data
        for result in percent_results:
            # Pad train_losses to max_epochs with NaN
            padded_train_losses = result['train_losses'] + [np.nan] * (max_epochs - len(result['train_losses']))
            all_train_losses.append(padded_train_losses)
            
            # Pad val_losses to max_epochs with NaN
            padded_val_losses = result['val_losses'] + [np.nan] * (max_epochs - len(result['val_losses']))
            all_val_losses.append(padded_val_losses)
            
            # Pad val_accs to max_epochs with NaN
            padded_val_accs = result['val_accs'] + [np.nan] * (max_epochs - len(result['val_accs']))
            all_val_accs.append(padded_val_accs)
            
            # Calculate relative times (seconds since start)
            if result['times']:
                start = result['times'][0]
                rel_times = [(t - start).total_seconds() for t in result['times']]
                # Pad rel_times to max_epochs with NaN
                padded_rel_times = rel_times + [np.nan] * (max_epochs - len(rel_times))
                all_rel_times.append(padded_rel_times)
                
                # Calculate total training time
                total_time = (result['times'][-1] - result['times'][0]).total_seconds()
                all_total_times.append(total_time)
            
            # Collect test accuracies
            if result['test_acc'] is not None:
                all_test_accs.append(result['test_acc'])
        
        # Convert to numpy arrays for calculations
        all_train_losses = np.array(all_train_losses)
        all_val_losses = np.array(all_val_losses)
        all_val_accs = np.array(all_val_accs)
        
        # Calculate means and standard deviations
        mean_train_losses = np.nanmean(all_train_losses, axis=0)
        std_train_losses = np.nanstd(all_train_losses, axis=0)
        
        mean_val_losses = np.nanmean(all_val_losses, axis=0)
        std_val_losses = np.nanstd(all_val_losses, axis=0)
        
        mean_val_accs = np.nanmean(all_val_accs, axis=0)
        std_val_accs = np.nanstd(all_val_accs, axis=0)
        
        # Store aggregated data
        aggregated = {
            'percent': percent,
            'epochs': list(range(1, max_epochs + 1)),
            'mean_train_losses': mean_train_losses,
            'std_train_losses': std_train_losses,
            'mean_val_losses': mean_val_losses,
            'std_val_losses': std_val_losses,
            'mean_val_accs': mean_val_accs,
            'std_val_accs': std_val_accs,
            'used_samples': percent_results[0]['used_samples'],  # Assuming consistent across runs
        }
        
        # Add time-related data if available
        if all_rel_times:
            all_rel_times = np.array(all_rel_times)
            mean_rel_times = np.nanmean(all_rel_times, axis=0)
            std_rel_times = np.nanstd(all_rel_times, axis=0)
            aggregated['mean_rel_times'] = mean_rel_times
            aggregated['std_rel_times'] = std_rel_times
        
        # Add test accuracy statistics if available
        if all_test_accs:
            all_test_accs = np.array(all_test_accs)
            mean_test_acc = np.mean(all_test_accs)
            std_test_acc = np.std(all_test_accs)
            aggregated['mean_test_acc'] = mean_test_acc
            aggregated['std_test_acc'] = std_test_acc
        
        # Add time spent statistics if available
        if all_total_times:
            all_total_times = np.array(all_total_times)
            mean_time_sec = np.mean(all_total_times)
            std_time_sec = np.std(all_total_times)
            aggregated['mean_time_sec'] = mean_time_sec
            aggregated['std_time_sec'] = std_time_sec
        
        aggregated_data[percent] = aggregated
    
    return aggregated_data

def parse_out_file(filepath):
    percent = int(re.search(r"fine_(\d+)\.out", filepath).group(1))
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    times = []
    used_samples = None
    total_samples = None
    test_acc = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    epoch_count = 0
    for line in lines:
        line = line.strip()
        if line.startswith("train Loss"):
            epoch_count += 1
            epochs.append(epoch_count)
            m = re.search(r"train Loss: ([\d.]+) Sample-Acc: ([\d.]+)", line)
            train_losses.append(float(m.group(1)))
        elif line.startswith("val Loss"):
            m = re.search(r"val Loss: ([\d.]+) Sample-Acc: ([\d.]+)", line)
            val_losses.append(float(m.group(1)))
            val_accs.append(float(m.group(2)))
        m2 = re.match(r"\[([\d\-]+\s[\d:]+)\] Finished epoch", line)
        if m2:
            times.append(datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S"))
        m3 = re.search(r"Loaded (\d+) total dog samples", line)
        if m3:
            total_samples = int(m3.group(1))
        m4 = re.search(r"Reduced training set size: (\d+)", line)
        if m4:
            used_samples = int(m4.group(1))
        m5 = re.search(r"Test Sample-Acc: ([\d.]+)", line)
        if m5:
            test_acc = float(m5.group(1))

    return {
        'percent': percent,
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'times': times,
        'used_samples': used_samples,
        'total_samples': total_samples,
        'test_acc': test_acc
    }

def save_summary_table(summary_df, filename):
    fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * len(summary_df)))
    ax.axis('off')
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_validation_loss(aggregated_data, percents, filename, show_std_dev=True):
    plt.figure(figsize=(10, 6))
    for percent in percents:
        if percent in aggregated_data:
            data = aggregated_data[percent]
            epochs = data['epochs']
            mean_val_losses = data['mean_val_losses']
            std_val_losses = data['std_val_losses']
            
            plt.plot(epochs, mean_val_losses, label=f"{percent}% Mean", linewidth=2)
            if show_std_dev:
                plt.fill_between(
                    epochs, 
                    mean_val_losses - std_val_losses,
                    mean_val_losses + std_val_losses, 
                    alpha=0.3
                )
    
    if plt.gca().has_data():
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    title = "Mean Validation Loss over Epochs"
    if show_std_dev:
        title += " with ±1 Std Dev"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_validation_accuracy(aggregated_data, percents, filename, show_std_dev=True):
    plt.figure(figsize=(10, 6))
    for percent in percents:
        if percent in aggregated_data:
            data = aggregated_data[percent]
            epochs = data['epochs']
            mean_val_accs = data['mean_val_accs']
            std_val_accs = data['std_val_accs']
            
            plt.plot(epochs, mean_val_accs, label=f"{percent}% Mean", linewidth=2)
            if show_std_dev:
                plt.fill_between(
                    epochs, 
                    mean_val_accs - std_val_accs,
                    mean_val_accs + std_val_accs, 
                    alpha=0.3
                )
    
    if plt.gca().has_data():
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    title = "Mean Validation Accuracy over Epochs"
    if show_std_dev:
        title += " with ±1 Std Dev"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_test_accuracy_vs_percent(summary_df, filename):
    plt.figure(figsize=(10, 6))
    
    # Extract data from summary dataframe
    percents = summary_df['percent']
    mean_test_accs = summary_df['mean_test_acc']
    std_test_accs = summary_df['std_test_acc']
    
    plt.errorbar(
        percents, 
        mean_test_accs, 
        yerr=std_test_accs, 
        marker='o', 
        linestyle='-', 
        linewidth=2,
        capsize=5
    )
    
    plt.xlabel("% Data Used")
    plt.ylabel("Test Accuracy")
    plt.title("Mean Test Accuracy vs % Data Used with ±1 Std Dev")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_val_loss_over_time(aggregated_data, percents, filename, show_std_dev=True):
    plt.figure(figsize=(10, 6))
    for percent in percents:
        if percent in aggregated_data and 'mean_rel_times' in aggregated_data[percent]:
            data = aggregated_data[percent]
            mean_rel_times = data['mean_rel_times']
            mean_val_losses = data['mean_val_losses']
            std_val_losses = data['std_val_losses']
            
            plt.plot(mean_rel_times, mean_val_losses, label=f"{percent}% Mean", linewidth=2)
            if show_std_dev:
                plt.fill_between(
                    mean_rel_times, 
                    mean_val_losses - std_val_losses,
                    mean_val_losses + std_val_losses, 
                    alpha=0.3
                )
    
    if plt.gca().has_data():
        plt.legend()
    plt.xlabel("Mean Seconds Since Start")
    plt.ylabel("Validation Loss")
    title = "Mean Validation Loss over Time"
    if show_std_dev:
        title += " with ±1 Std Dev"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_training_loss(aggregated_data, percents, filename, show_std_dev=True):
    plt.figure(figsize=(10, 6))
    for percent in percents:
        if percent in aggregated_data:
            data = aggregated_data[percent]
            epochs = data['epochs']
            mean_train_losses = data['mean_train_losses']
            std_train_losses = data['std_train_losses']
            
            plt.plot(epochs, mean_train_losses, label=f"{percent}% Mean", linewidth=2)
            if show_std_dev:
                plt.fill_between(
                    epochs, 
                    mean_train_losses - std_train_losses,
                    mean_train_losses + std_train_losses, 
                    alpha=0.3
                )
    
    if plt.gca().has_data():
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    title = "Mean Training Loss over Epochs"
    if show_std_dev:
        title += " with ±1 Std Dev"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process output files from multiple directories and generate plots with means and standard deviations.")
    parser.add_argument('dirs', nargs='+', help='Directories containing fine_*.out files to process')
    args = parser.parse_args()
    
    # Dictionary to hold results from all directories
    all_results = {}
    
    # Process files from each directory
    for dir_name in args.dirs:
        if not os.path.isdir(dir_name):
            print(f"Warning: Directory '{dir_name}' does not exist. Skipping.")
            continue
            
        search_pattern = os.path.join(dir_name, "fine_*.out")
        filepaths = glob.glob(search_pattern)
        
        if not filepaths:
            print(f"No fine_*.out files found in directory '{dir_name}'. Skipping.")
            continue
            
        # Sort files by percentage
        sorted_filepaths = sorted(filepaths, 
                                key=lambda x: int(re.search(r"fine_(\d+)\.out", os.path.basename(x)).group(1)))
        
        # Parse all files in this directory
        all_results[dir_name] = [parse_out_file(fp) for fp in sorted_filepaths]
        print(f"Processed {len(sorted_filepaths)} files from '{dir_name}'.")
    
    if not all_results:
        print("No valid directories with fine_*.out files found.")
    else:
        # Aggregate results across all directories
        aggregated_data = aggregate_results(all_results)
        
        # Prepare summary data
        summary_records = []
        for percent, data in aggregated_data.items():
            if 'mean_test_acc' in data and 'std_test_acc' in data and 'mean_time_sec' in data and 'std_time_sec' in data:
                # Format test accuracy with mean and std dev
                test_acc_formatted = f"{data['mean_test_acc']:.4f} ± {data['std_test_acc']:.4f}"
                
                # Format time spent with mean and std dev
                mean_time_formatted = format_time(data['mean_time_sec'])
                std_time_sec = data['std_time_sec']
                time_spent_formatted = f"{mean_time_formatted} ± {std_time_sec:.1f}s"
                
                summary_records.append({
                    'percent': percent,
                    'used_samples': data['used_samples'],
                    'mean_test_acc': data['mean_test_acc'],
                    'std_test_acc': data['std_test_acc'],
                    'test_acc': test_acc_formatted,
                    'time_spent': time_spent_formatted
                })
        
        # Create and display summary DataFrame
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_df = summary_df.sort_values('percent')
            
            # Select columns for display
            display_df = summary_df[['percent', 'used_samples', 'test_acc', 'time_spent']]
            display_df.columns = ['percent', 'used_samples', 'test_acc (mean ± std)', 'time_spent (mean ± std)']
            
            print("\nSummary of results across all directories:")
            print(display_df)
            
            # Save summary table as PNG
            save_summary_table(display_df, "avg_summary_table.png")
            
            # Create plotting dataframe with numeric columns for the test accuracy plot
            plot_df = summary_df[['percent', 'mean_test_acc', 'std_test_acc']]
            
            # Generate and save plots without standard deviation shading for all percentages
            percents_to_plot = [1, 10, 20, 40, 60, 80, 100]
            plot_validation_loss(aggregated_data, percents_to_plot, "avg_val_loss_epochs.png", show_std_dev=False)
            plot_validation_accuracy(aggregated_data, percents_to_plot, "avg_val_accuracy_epochs.png", show_std_dev=False)
            plot_val_loss_over_time(aggregated_data, percents_to_plot, "avg_val_loss_time.png", show_std_dev=False)
            plot_training_loss(aggregated_data, percents_to_plot, "avg_train_loss_epochs.png", show_std_dev=False)
            
            # Generate and save plots with standard deviation shading for subset of percentages (1%, 40%, 100%)
            percents_for_shaded = [1, 40, 80, 100]
            plot_validation_loss(aggregated_data, percents_for_shaded, "avg_val_loss_epochs_shaded_subset.png", show_std_dev=True)
            plot_validation_accuracy(aggregated_data, percents_for_shaded, "avg_val_accuracy_epochs_shaded_subset.png", show_std_dev=True)
            plot_val_loss_over_time(aggregated_data, percents_for_shaded, "avg_val_loss_time_shaded_subset.png", show_std_dev=True)
            
            print("\nAggregated plots and summary table saved as PNG files:")
            print("  - Standard plots (all percentages, no standard deviation shading): avg_*")
            print("  - Subset plots (1%, 40%, 100% with standard deviation shading): avg_*_shaded_subset.png")
        else:
            print("No complete data found to generate summary and plots.")


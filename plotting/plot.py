import re
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    """Parses the training log file to extract metrics."""
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
    train_pattern = re.compile(r"train Loss: ([\d.]+) Sample-Acc: ([\d.]+)")
    val_pattern = re.compile(r"val Loss: ([\d.]+) Sample-Acc: ([\d.]+)")

    current_epoch = None

    with open(log_file_path, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
                # Reset temp holders for the new epoch
                current_train_loss = None
                current_train_acc = None
                current_val_loss = None
                current_val_acc = None

            train_match = train_pattern.search(line)
            if train_match and current_epoch is not None:
                current_train_loss = float(train_match.group(1))
                current_train_acc = float(train_match.group(2))

            val_match = val_pattern.search(line)
            if val_match and current_epoch is not None:
                current_val_loss = float(val_match.group(1))
                current_val_acc = float(val_match.group(2))

            # Check if we have collected all data for the current epoch
            # This assumes train and val lines appear after the Epoch line for that epoch
            if current_epoch is not None and current_train_loss is not None and current_val_loss is not None:
                 # Check if the current epoch's data has already been added
                 # This handles cases where metrics might appear before the "Finished epoch" line
                 if len(train_losses) < current_epoch:
                    train_losses.append(current_train_loss)
                    train_accs.append(current_train_acc)
                    val_losses.append(current_val_loss)
                    val_accs.append(current_val_acc)
                    # Reset temp holders after adding
                    current_train_loss = None
                    current_train_acc = None
                    current_val_loss = None
                    current_val_acc = None


    # Ensure all lists have the same length, trim epochs if necessary
    min_len = min(len(train_losses), len(train_accs), len(val_losses), len(val_accs))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    train_accs = train_accs[:min_len]
    val_losses = val_losses[:min_len]
    val_accs = val_accs[:min_len]


    return epochs, train_losses, train_accs, val_losses, val_accs

def plot_metrics(epochs, train_losses, train_accs, val_losses, val_accs, loss_filename="training_loss_plot.png", acc_filename="training_accuracy_plot.png"):
    """Plots training and validation loss and accuracy into separate files."""
    # Plot Loss
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(loss_filename)
    print(f"Loss plot saved to {loss_filename}")
    plt.close(fig1) # Close the figure to free memory

    # Plot Accuracy
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(acc_filename)
    print(f"Accuracy plot saved to {acc_filename}")
    plt.close(fig2) # Close the figure to free memory
    # plt.show() # Uncomment to display the plots interactively

if __name__ == "__main__":
    log_file = 'base_model.out'
    epochs, train_losses, train_accs, val_losses, val_accs = parse_log_file(log_file)

    if not epochs:
        print("No data parsed from the log file. Please check the file format and content.")
    else:
        print(f"Parsed data for {len(epochs)} epochs.")
        print("Epochs:", epochs)
        print("Train Losses:", train_losses)
        print("Train Accs:", train_accs)
        print("Val Losses:", val_losses)
        print("Val Accs:", val_accs)
        plot_metrics(epochs, train_losses, train_accs, val_losses, val_accs)

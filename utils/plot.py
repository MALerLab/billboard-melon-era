from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
from io import BytesIO
from tqdm import tqdm
import numpy as np
import torch


def draw_bar_graph(info, base_fig_size = (10, 8), base_font_size = 20, RATIO = 1, color = 'skyblue'):
    """
    info example:

    hv_histogram_info = {
        'xlabel': 'Difficulty Level',
        'ylabel': 'Number of Pieces',
        'title': 'Difficulty Distribution of the Hidden Voices Dataset',
        'save_path': 'images/hv_difficulty_distribution.png',
        'category': [1, 2, 3, 4, 5],
        'distribution': [10, 20, 30, 40, 50]
    }
    """

    # Scale figure size
    fig_size = (base_fig_size[0] * RATIO, base_fig_size[1] * RATIO)
    # Scale font sizes
    font_size = base_font_size * RATIO
    title_font_size = font_size * 1.14  # Title font size slightly larger
    bar_text_font_size = font_size * 0.85  # Text on bars slightly smaller

    # Create figure with scaled size
    plt.figure(figsize=fig_size)
    plt.xticks(range(len(info['category'])), info['category'], fontsize=bar_text_font_size)
    plt.yticks(fontsize=bar_text_font_size)
    # Set labels with scaled font size and padding
    plt.xlabel(info['xlabel'], fontsize=font_size, labelpad=font_size)
    plt.ylabel(info['ylabel'], fontsize=font_size, labelpad=font_size)
    # Set title with scaled font size
    plt.title(info['title'], fontsize=title_font_size, pad=font_size)
    # plt.gca().title.set_position([0.5, 1.05])  # Set title position at the top
    # Add value per bar at the top
    for i, v in enumerate(info['distribution']):
        plt.text(i, v, str(v), ha='center', va='bottom', fontsize=bar_text_font_size)
    
    plt.bar(info['category'], info['distribution'], color=color)
    if 'save_path' in info.keys():
        Path(info['save_path']).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(info['save_path'], bbox_inches='tight')
    plt.show()


def concatenate_images(image_paths, save_path, x_space = 0):
    # Open the images
    images = [Image.open(image_path) for image_path in image_paths]

    # Get the maximum height of the two images
    max_height = max(image.size[1] for image in images)

    # Create a new image with the combined width and the maximum height
    total_width = sum(image.size[0] + x_space for image in images)
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the images into the new image
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]
        empty_image = Image.new('RGB', (x_space, max_height), color='white')
        new_image.paste(empty_image, (x_offset, 0))
        x_offset += x_space

    # Save the new image
    new_image.save(save_path)


def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save as an image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    image = Image.open(buf)

    return image

def plot_umap_in_train(features, labels):
    # Use only the labels of the last hierarchy level
    labels = labels[:, -1]
    
    # Label encoding
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Count the number of samples in each class
    unique, counts = np.unique(labels_encoded, return_counts=True)
    min_samples = min(counts)  # find the smallest sample count
    
    # Randomly draw the same number of samples from every class
    resampled_features = []
    resampled_labels = []
    for i in unique:
        idx = np.where(labels_encoded == i)[0]
        resampled_idx = resample(idx, n_samples=min_samples, replace=False, random_state=42)
        resampled_features.extend(features[resampled_idx])
        resampled_labels.extend(labels_encoded[resampled_idx])
    
    # Create the UMAP reducer
    import umap.umap_ as umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    # Apply UMAP
    umap_results = reducer.fit_transform(resampled_features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=resampled_labels, cmap='Spectral', s=3)
    plt.colorbar()
    plt.title('UMAP projection of the Training Set')
    # Create a buffer to save the image into
    buf = BytesIO()
    plt.savefig(buf, format='png')
    # Rewind the buffer to its start
    buf.seek(0)
    # Load the image
    image = Image.open(buf)
    # Convert to a NumPy array
    image_np = np.array(image)
    # Release resources
    plt.close()
    buf.close()
    
    return image_np


def plot_umap(model, test_loader, DEV):
    # TODO: remove duplicated parts
    def _extract_features_and_labels(model, data_loader, device):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for audio, label, _ in tqdm(data_loader):
                audio = audio.to(device)
                feature = model(audio, return_embedding=True).cpu().numpy()
                features.append(feature)
                labels.extend(label)
        return np.concatenate(features), np.array(labels)

    features, labels = _extract_features_and_labels(model, test_loader, DEV)
    import umap.umap_ as umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_results = reducer.fit_transform(features)
    
    # Render the UMAP plots
    for s_val in [3, 10, 30, 50]:
        plt.figure(figsize=(12, 10))
        plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='Spectral', s=s_val)
        plt.colorbar()
        plt.title('UMAP projection of the Test Set')
        plt.savefig(f'testset_umap_projection{s_val}.png')
        plt.close()


def plot_individual_confusion_matrices(all_labels_hierarchy, all_preds_hierarchy, class_names):
    images = []
    total_hierarchies = len(all_labels_hierarchy)
    for i, (labels, preds, class_name) in enumerate(zip(all_labels_hierarchy, all_preds_hierarchy, class_names)):
        cm = confusion_matrix(labels, preds, normalize='true')
        fig, ax = plt.subplots(figsize=(14, 14))  
        if i == total_hierarchies - 1:
            sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_name, yticklabels=class_name, ax=ax)
        else:
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_name, yticklabels=class_name, ax=ax)
        plt.title(f'Hierarchy {i} Confusion Matrix', size=20)
        plt.ylabel('True Label', size=16)
        plt.xlabel('Predicted Label', size=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout() 
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_np = np.array(image)
        images.append(image_np)
        plt.close(fig)
        buf.close()
    return images



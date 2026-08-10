
from pathlib import Path
import numpy as np
import csv
import json
from tqdm import tqdm
from typing import List, Dict, Union
from PIL import Image
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_zoo import  Baseline, FCN, ShortChunkCNN, ShortChunkCNN_Res, Musicnn, CRNN
from dataset import BillboardDatasetHierarchyTrain
from utils.plot import plot_individual_confusion_matrices, plot_umap_in_train


class Trainer:
  def __init__(self,
              model: Union[Baseline, FCN, ShortChunkCNN, ShortChunkCNN_Res, Musicnn, CRNN], 
              optimizer: torch.optim.Optimizer,
              loss_func: nn.Module,
              validation_loader: DataLoader,
              iterations_per_chunk: int,
              device: str = 'cuda',
              validation_freq: int = 500,
              collate_fn: callable = None,
              save_dir: str = 'models',
              best_model_dir: str = 'best_model',
              hierarchy_class_map: List[Dict[int, list]] = None,
              high_to_low_weight: float = 0.1,
              low_to_high_weight: float = 0.1,
              config: Dict[str, Union[int, float, str]] = None,
              adv_loader: DataLoader = None,
              test_loader: DataLoader = None,
              ):
    
    self.model = model
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.validation_loader = validation_loader
    self.adv_loader = adv_loader
    self.iterations_per_chunk = iterations_per_chunk
    self.device = device
    self.validation_freq = validation_freq
    self.collate_fn = collate_fn
    self.save_dir = Path(save_dir)
    self.best_model_dir = Path(best_model_dir)

    self.num_updated = 0
    self.high_to_low_weight = high_to_low_weight
    self.low_to_high_weight = low_to_high_weight
    self.hierarchy_class_map = hierarchy_class_map
    
    self.config = config
    
    self.best_valid_macro = 0
    # The paper selects on lowest validation loss, the original code selected on highest
    # validation macro accuracy. Track both and keep a checkpoint for each, so the
    # criterion can be settled after the run instead of being baked into it.
    self.best_valid_loss = float('inf')
    self.best_selection = {}
    self.test_loader = test_loader
    self.test_iterations = 1000 
    
    self.class_names = [['WHOLE_1960', 'WHOLE_1970', 'WHOLE_1980', 'WHOLE_1990', 'WHOLE_2000', 'WHOLE_2010'],
                        ['HALF_1960_1', 'HALF_1960_2', 'HALF_1970_1', 'HALF_1970_2', 'HALF_1980_1', 'HALF_1980_2',
                        'HALF_1990_1', 'HALF_1990_2', 'HALF_2000_1', 'HALF_2000_2', 'HALF_2010_1', 'HALF_2010_2'],
                        ['QUARTER_1960_1', 'QUARTER_1960_2', 'QUARTER_1960_3', 'QUARTER_1960_4', 
                        'QUARTER_1970_1', 'QUARTER_1970_2', 'QUARTER_1970_3', 'QUARTER_1970_4', 
                        'QUARTER_1980_1', 'QUARTER_1980_2', 'QUARTER_1980_3', 'QUARTER_1980_4', 
                        'QUARTER_1990_1', 'QUARTER_1990_2', 'QUARTER_1990_3', 'QUARTER_1990_4', 
                        'QUARTER_2000_1', 'QUARTER_2000_2', 'QUARTER_2000_3', 'QUARTER_2000_4',
                        'QUARTER_2010_1', 'QUARTER_2010_2', 'QUARTER_2010_3', 'QUARTER_2010_4'],
                        [str(year) for year in range(1958, 2025)]]
        
  # Training
  def _train_with_single_batch(self, batch):
    train_loss = torch.tensor(0., device=self.device)
    self.optimizer.zero_grad()
    if len(batch) == 2:
      audio, labels = batch
    elif len(batch) == 3:
      audio, labels, adv_labels = batch
    else:
      raise ValueError('Batch Type must be 2 or 3')
    audio = audio.to(self.device)
    preds:List[torch.Tensor] = self.model(audio)
    train_acc_hierarchy = []
    loss_dict = {}
    
    for hierarchy in range(4):
      label = torch.tensor([lbl[hierarchy] for lbl in labels], device=self.device)
      pred = preds[hierarchy]
      if len(batch) == 3: pred = pred[:len(label)]
      loss = self.loss_func(pred, label)
      train_loss += loss
      acc = (pred.argmax(dim=-1) == label).float().mean().item() * 100
      train_acc_hierarchy.append(acc)
      loss_dict[f'Hierarchy {hierarchy} Loss'] = loss.item()
    
    if len(batch) == 3: # including domain classification:
      domain_label = adv_labels
      domain_loss = nn.functional.binary_cross_entropy(preds[-1].squeeze(-1), domain_label.to(self.device))
      train_loss += domain_loss * self.config.train.adversarial_loss_weight
      loss_dict['Domain Classification Loss'] = domain_loss.item()
      
    high_to_low_loss, low_to_high_loss = self._get_hierarchy_consistency_loss([pred[:len(labels[0])] for pred in preds])
    loss_dict['High to Low Loss'] = high_to_low_loss.item()
    loss_dict['Low to High Loss'] = low_to_high_loss.item()
    # hierarchy_consistency = self._check_hierarchical_consistency(preds)
    
    train_loss /= 4
    train_loss += high_to_low_loss * self.high_to_low_weight + low_to_high_loss * self.low_to_high_weight
    return train_loss, train_acc_hierarchy, loss_dict
  
  # Hierarchy Consistency
  def _get_hierarchy_consistency_loss(self, preds:List[torch.Tensor]):
    if self.high_to_low_weight == 0 and self.low_to_high_weight == 0:
      return torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
    
    # hierarchy consistency loss
    high_to_low_loss = torch.tensor(0., device=self.device)
    low_to_high_loss = torch.tensor(0., device=self.device)
    for i in range(3):
      high_pred = preds[i]
      low_pred = preds[i+1]
      
      for class_idx in range(high_pred.shape[1]):
        high_class_prob = high_pred[:, class_idx]
        corresponding_low_class_ids = self.hierarchy_class_map[i][class_idx]
        low_class_prob = low_pred[:, corresponding_low_class_ids]
        low_to_high_loss += (torch.clamp_min(low_class_prob - high_class_prob.unsqueeze(1), 0) ** 2).sum()
        high_to_low_loss += (torch.clamp_min(high_class_prob - low_class_prob.max(dim=-1).values, 0) ** 2).sum()
    
    num_classes = sum([pred.shape[1] for pred in preds[:-1]])
    num_low_classes = sum([pred.shape[1] for pred in preds[1:]])
    low_to_high_loss /= num_low_classes
    high_to_low_loss /= num_classes
    
    return high_to_low_loss, low_to_high_loss
  
  def _check_hierarchical_consistency(self, preds: List[torch.Tensor]):
    consistent_preds_count = {hierarchy: 0 for hierarchy in range(3)}
    for hierarchy in range(3):
      for sample in range(preds[hierarchy].shape[0]):
        parents_pred = preds[hierarchy][sample].argmax(dim=-1).item()
        child_pred = preds[hierarchy+1][sample].argmax(dim=-1).item()
        if child_pred in self.hierarchy_class_map[hierarchy][parents_pred]:
          consistent_preds_count[hierarchy] += 1
    hierarchy_consistency = {hierarchy: (count / preds[hierarchy].shape[0])*100 for hierarchy, count in consistent_preds_count.items()}
    return hierarchy_consistency
  
  
  # Validation
  def _validate(self):
    valid_loss, valid_avg_acc, valid_acc_hierarchy, hierarchy_consistency_scores, valid_all_preds, valid_all_labels  = self._evaluate_model()
    validation_metric_dict = self.calculate_metrics(valid_all_preds, valid_all_labels)
    wandb.log({
                "Validation Loss": valid_loss,
                "Validation Accuracy": valid_avg_acc,
                **{f"Validation Layer {i+1} Accuracy": acc for i, acc in enumerate(valid_acc_hierarchy)},
                **{f"Validation Hierarchy {i} Consistency": acc for i, acc in hierarchy_consistency_scores.items()}
            }, step=self.num_updated)
    
    if self.best_valid_macro < validation_metric_dict['MACRO ACCURACY']:
      self.best_valid_macro = validation_metric_dict['MACRO ACCURACY']
      torch.save(self.model.state_dict(), self.best_model_dir/f'macro_{self.num_updated}.pt')
      self.best_selection['macro'] = {'iteration': self.num_updated,
                                      'valid_macro_accuracy': self.best_valid_macro,
                                      'valid_loss': float(valid_loss)}
      self.is_improved = True
    else:
      self.is_improved = False

    if valid_loss < self.best_valid_loss:
      self.best_valid_loss = float(valid_loss)
      torch.save(self.model.state_dict(), self.best_model_dir/f'loss_{self.num_updated}.pt')
      self.best_selection['loss'] = {'iteration': self.num_updated,
                                     'valid_macro_accuracy': validation_metric_dict['MACRO ACCURACY'],
                                     'valid_loss': self.best_valid_loss}

    with open(self.best_model_dir/'best_selection.json', 'w') as f:
      json.dump(self.best_selection, f, indent=1)


    valid_acc_hierarchy_str = ", ".join([f"Layer {i+1}: {acc:.2f}%" for i, acc in enumerate(valid_acc_hierarchy)])
    print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_avg_acc:.2f}%, Layer Accuracies: [{valid_acc_hierarchy_str}]')
  
  def _evaluate_model(self, valid_test_loader: DataLoader = None):
    if valid_test_loader is None:
      valid_test_loader = self.validation_loader
    self.model.eval() 
    valid_loss = 0
    valid_acc_hierarchy = [0 for _ in range(4)]
    all_preds_hierarchy, all_labels_hierarchy, sample_count_hierarchy = [[] for _ in range(4)], [[] for _ in range(4)], [0 for _ in range(4)]
    hierarchy_consistency_accum = {hierarchy: 0 for hierarchy in range(3)}
    num_batches = 0
    
    with torch.inference_mode():
      for batch in valid_test_loader:
        audio, labels = batch
        audio = audio.to(self.device)
        preds = self.model(audio)
        
        for hierarchy in range(4):
          label = torch.tensor([lbl[hierarchy] for lbl in labels], device=self.device)
          pred = preds[hierarchy]
          loss = self.loss_func(pred, label)
          valid_loss += loss.item() * len(label)
          valid_acc_hierarchy[hierarchy] += (pred.argmax(dim=-1) == label).float().sum().item()
          
          sample_count_hierarchy[hierarchy] += len(label)
          all_preds_hierarchy[hierarchy].extend(pred.argmax(dim=-1).cpu().numpy())
          all_labels_hierarchy[hierarchy].extend(label.cpu().numpy())
          
        batch_hierarchy_consistency = self._check_hierarchical_consistency(preds)
        for hierarchy in range(3):
          hierarchy_consistency_accum[hierarchy] += batch_hierarchy_consistency[hierarchy]
            
        num_batches += 1
        
      valid_loss /= sum(sample_count_hierarchy)
      valid_avg_acc = sum(valid_acc_hierarchy) / sum(sample_count_hierarchy) * 100
      hierarchy_consistency_scores = {hierarchy: score / num_batches for hierarchy, score in hierarchy_consistency_accum.items()}
      
    self.model.train()  
    return valid_loss, valid_avg_acc, [acc / count * 100 for acc, count in zip(valid_acc_hierarchy, sample_count_hierarchy)], hierarchy_consistency_scores, all_preds_hierarchy, all_labels_hierarchy
  
  # Metrics
  def _calculate_MSE(self, h_preds, h_labels):
    mse = ((np.array(h_preds) - np.array(h_labels)) ** 2).mean()
    return mse
  
  def _calculate_MAE(self, h_preds, h_labels):
    mae = np.abs(np.array(h_preds) - np.array(h_labels)).mean()
    return mae
  
  def calculate_metrics(self, all_preds, all_labels):
    metrics_dict = {}
    for h_idx, class_name_list in enumerate(self.class_names): # class_name_list example: ['WHOLE_1960', 'WHOLE_1970', 'WHOLE_1980', 'WHOLE_1990', 'WHOLE_2000', 'WHOLE_2010'],
      h_metrics_dict = {}
      h_preds = all_preds[h_idx]
      h_labels = all_labels[h_idx]
      total_correct, total_preds = 0, 0
      for c_idx, class_name in enumerate(class_name_list):
        c_preds = [pred for pred, label in zip(h_preds, h_labels) if label == c_idx]
        correct_preds = sum(pred == c_idx for pred in c_preds)
        total_correct += correct_preds
        total_preds += len(c_preds)
        if len(c_preds) > 0:
          c_acc = correct_preds / len(c_preds)
        else:
          c_acc = 0
        h_metrics_dict[class_name] = c_acc
      h_metrics_dict['macro accuracy'] = np.mean(list(h_metrics_dict.values()))
      h_metrics_dict['micro accuracy'] = total_correct / total_preds
      h_metrics_dict['mse'] = self._calculate_MSE(h_preds, h_labels)
      h_metrics_dict['mae'] = self._calculate_MAE(h_preds, h_labels)
      metrics_dict[f'Hierarchy{h_idx}'] = h_metrics_dict
    metrics_dict['MACRO ACCURACY'] = np.mean([h_metrics_dict['macro accuracy'] for h_metrics_dict in metrics_dict.values()])
    return metrics_dict
  
  def _test(self, is_improved: bool = False):
    if self.test_loader is None:
      return
    self.model.eval()
    all_preds_hierarchy, all_labels_hierarchy, test_loss, test_acc, hierarchy_consistency_scores = self._test_model(self._nll_loss_individual, self.device)
    if is_improved:
      metrics_dict = self.calculate_metrics(all_preds_hierarchy, all_labels_hierarchy)
      metrics_dict['hierarchy_consistency'] = hierarchy_consistency_scores
      # save json
      with open(self.best_model_dir/f'{self.num_updated}_test_metrics.json', 'w') as f:
        json.dump(metrics_dict, f)
      
    individual_confusion_matrix_images = plot_individual_confusion_matrices(all_preds_hierarchy, all_labels_hierarchy, self.class_names)
    wandb.log({
                "Test Loss": test_loss, 
                **{f'Test Accuracy Hierarchy {i}': acc for i, acc in enumerate(test_acc)},
                **{f"Test Hierarchy {i} Consistency": acc for i, acc in hierarchy_consistency_scores.items()},
                **{f'Confusion Matrix Hierarchy {i}': wandb.Image(Image.fromarray(image_np)) for i, image_np in enumerate(individual_confusion_matrix_images)},
            }, step=self.num_updated)
  
  def _nll_loss_individual(self, pred, target):
    loss_per_sample = -torch.log(pred[range(len(target)), target])
    return loss_per_sample
  
  def _test_model(self, loss_func, device):
    self.model.to(device)
    self.model.eval()
    all_preds_hierarchy, all_labels_hierarchy, sample_count_hierarchy = [[] for _ in range(4)], [[] for _ in range(4)], [0 for _ in range(4)]
    test_acc_hierarchy = [0 for _ in range(4)]
    test_loss = 0
    hierarchy_consistency_accum = {hierarchy: 0 for hierarchy in range(3)}
    num_batches = 0
        
    with torch.inference_mode():
      for batch in self.test_loader:
        audio, labels, file_names = batch
        audio = audio.to(device)
        preds = self.model(audio)
        for hierarchy in range(4):
          label = torch.tensor([lbl[hierarchy] for lbl in labels], device=self.device)
          pred = preds[hierarchy]
          loss = loss_func(pred, label)
          test_loss += loss.mean().item() * len(label)
          
          correct_pred = (pred.argmax(dim=1) == label).float().sum().item()
          test_acc_hierarchy[hierarchy] += correct_pred
          sample_count_hierarchy[hierarchy] += len(label)
          all_preds_hierarchy[hierarchy].extend(pred.argmax(dim=-1).cpu().numpy())
          all_labels_hierarchy[hierarchy].extend(label.cpu().numpy())

        batch_hierarchy_consistency = self._check_hierarchical_consistency(preds)
        for hierarchy in range(3):
          hierarchy_consistency_accum[hierarchy] += batch_hierarchy_consistency[hierarchy]
            
        num_batches += 1

    test_loss /= sum(sample_count_hierarchy)
    test_acc = [acc / count * 100 for acc, count in zip(test_acc_hierarchy, sample_count_hierarchy)]
    hierarchy_consistency_scores = {hierarchy: score / num_batches for hierarchy, score in hierarchy_consistency_accum.items()}
    self.model.train()
    
    return all_preds_hierarchy, all_labels_hierarchy, test_loss, test_acc, hierarchy_consistency_scores
  
  
  def train_chunk(self, chunk_loader: DataLoader, adv_cycles: int = 4):
    self.model.to(self.device)
    current_iteration = 0

    train_acc_hierarchy = [[] for _ in range(4)]
    print(f'current_iteration: {current_iteration}, iterations_per_chunk: {self.iterations_per_chunk}')
    pbar = tqdm(total=self.iterations_per_chunk, desc='Training')
    
    adv_iter = iter(self.adv_loader) if self.adv_loader is not None else None
    
    while current_iteration < self.iterations_per_chunk:
      self.model.train()
      # One pass over the loader is one epoch: re-draw the decade-balanced subset here so
      # the undersampling is re-randomized per epoch, as described in the paper. Workers
      # are re-forked when the iterator restarts, so they pick up the new subset.
      if hasattr(chunk_loader.dataset, 'resample_epoch'):
        chunk_loader.dataset.resample_epoch()
      for batch in chunk_loader:
        if current_iteration >= self.iterations_per_chunk: break
        if self.adv_loader is not None and current_iteration % adv_cycles == 0:
          try:
            adv_batch = next(adv_iter)
          except StopIteration:
            adv_iter = iter(self.adv_loader)
            adv_batch = next(adv_iter)
          audio, labels = batch
          adv_audio, adv_labels = adv_batch
          pop_label = torch.ones(len(labels))
          kpop_label = torch.zeros(len(adv_labels))
          batch = (torch.cat([audio, adv_audio], dim=0), labels, torch.cat([pop_label, kpop_label], dim=0))
          
        train_loss, train_acc_hierarchy, loss_dict = self._train_with_single_batch(batch)
        train_loss.backward()
        self.optimizer.step()

        avg_train_acc = sum(train_acc_hierarchy) / 4
        
        wandb.log({
                  "Train Loss": train_loss.item(),
                  "Average Training Accuracy": avg_train_acc,
                  **{f"Layer {i+1} Accuracy": acc for i, acc in enumerate(train_acc_hierarchy)},
                  **loss_dict,
                  # **{f"Hierarchy {i} Consistency": acc for i, acc in hierarchy_consistency.items()}
              }, step=self.num_updated)
        
        pbar.update(1)
        pbar.set_postfix({'Train Loss': f'{train_loss.item():.4f}', 'Average Training Accuracy': f'{avg_train_acc:.2f}%'})

        if self.num_updated % self.validation_freq == 0:
          self._validate()
          torch.save(self.model.state_dict(), self.save_dir / f'model_{self.num_updated}.pt')
          
        if self.num_updated % self.test_iterations == 0:
          self._test(is_improved=self.is_improved)
        
            
        current_iteration += 1
        self.num_updated += 1   

    pbar.close()
    
  def _get_min_sample_count(self,
                            json_path: str = ''):
    # decade_counts = {decade: 0 for decade in range(1960, 2020, 10)}
    decade_counts = {decade: 0 for decade in range(1960, 2010, 10)}
    with open(json_path, 'r') as f:
      train_data = json.load(f)['train']
    for file in train_data:
      year = file.split('_')[0]
      year = int(year[1:-1])
      if year < 1960:
        decade = 1960
      elif year >= 2020:
        decade = 2010
      else:
        decade = year - (year % 10)
      decade_counts[decade] += 1
    min_sample_count = min(decade_counts.values())
    return min_sample_count * len(decade_counts)
  
  def train_model_with_chunks(self, 
                              manual_seed: int = 77, 
                              json_path: str = '', 
                              cycles: int = 1,
                              num_chunks: int = 1):
    # Use the run's configured seed so different-seed runs actually diverge here;
    # falls back to the historical default (77) when no seed is configured.
    torch.manual_seed(int(self.config.train.get('seed', manual_seed)))

    # The dataset holds every track of the split whole and draws both the 30-second crop
    # and the decade-balanced subset per epoch, so there is nothing left to chunk here.
    trainset = BillboardDatasetHierarchyTrain(**self.config.train_set,
                                              chunk_indices=[],
                                              mode='train')
    print(f'Tracks resident: {len(trainset.loaded_files)}, samples per epoch: {len(trainset)}')
    train_loader = DataLoader(trainset,
                              batch_size=self.config.train.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=self.collate_fn)
    self.train_chunk(train_loader, adv_cycles=4) # How often to train with adversarial data
    del train_loader, trainset

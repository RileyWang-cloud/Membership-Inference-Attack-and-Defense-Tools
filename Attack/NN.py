import numpy as np
import math
import os

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NNAttack(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NNAttack, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size//2)
            self.fc3 = nn.Linear(hidden_size//2, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            out = self.sigmoid(out)
            return out


class TagByNNMIA (object):
    def __init__(self, shadow_train_performance, shadow_test_performance, num_classes, 
                 batch_size=128, hidden_size=32, model_dir="attack_models"):
        self.num_classes = num_classes
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.model_dir = os.path.join(model_dir, "nn")
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_attack_model(self, num_epochs=100, verbose=False):

        for class_idx in range(self.num_classes):

            model_path = os.path.join(self.model_dir, f"attack_model_class_{class_idx}.pth")

            if not os.path.exists(model_path):
                member_mask = (self.s_tr_labels == class_idx).reshape(-1, )
                non_member_mask = (self.s_te_labels == class_idx).reshape(-1, )
                
                member_X_train = np.column_stack((self.s_tr_outputs[member_mask], self.s_tr_labels[member_mask]))
                non_member_X_train = np.column_stack((self.s_te_outputs[non_member_mask], self.s_te_labels[non_member_mask]))
                
                attack_X_train = np.vstack((member_X_train, non_member_X_train))
                attack_y_train = np.hstack((np.ones(len(member_X_train)), np.zeros(len(non_member_X_train))))
                
                attack_X_train, attack_X_val, attack_y_train, attack_y_val = train_test_split(attack_X_train, attack_y_train, test_size=0.2, random_state=42, stratify=attack_y_train)

                print(f"Training attack model for class {class_idx}")
                print("Training data shape:", attack_X_train.shape)

                attack_X_train = torch.tensor(attack_X_train, dtype=torch.float32)
                attack_y_train = torch.tensor(attack_y_train, dtype=torch.float32).unsqueeze(1)
                attack_X_val = torch.tensor(attack_X_val, dtype=torch.float32)
                attack_y_val = torch.tensor(attack_y_val, dtype=torch.float32).unsqueeze(1)
                
                train_dataset = TensorDataset(attack_X_train, attack_y_train)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_dataset = TensorDataset(attack_X_val, attack_y_val)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                
                model = NNAttack(attack_X_train.shape[-1], self.hidden_size, 1)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                

                # 初始化early stopping参数
                patience = 10  # 如果验证集性能在patience个epoch内没有提升，则停止训练
                best_val_loss = float('inf')
                epochs_no_improve = 0

                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                    
                    epoch_loss = running_loss / len(train_loader.dataset)
                    
                    # 计算验证集上的损失
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for val_inputs, val_labels in val_loader:  # 假设你有一个val_loader
                            val_outputs = model(val_inputs)
                            val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)
                        val_loss /= len(val_loader.dataset)
                    
                    if verbose:
                        print(f'Class {class_idx} Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
                    
                    # 检查early stopping条件
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), model_path)  # 保存当前最优模型
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print(f'Early stopping at epoch {epoch+1}')
                            break

    def perform_attack(self, target_train_performance, target_test_performance, eval_metric='accuracy'):
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        all_preds, all_labels, all_preds_probs = [], [], []
        
        for class_idx in range(self.num_classes):
            
            member_mask = (self.t_tr_labels == class_idx).reshape(-1, )
            non_member_mask = (self.t_te_labels == class_idx).reshape(-1, )
            
            member_X_test = np.column_stack((self.t_tr_outputs[member_mask], self.t_tr_labels[member_mask]))
            non_member_X_test = np.column_stack((self.t_te_outputs[non_member_mask], self.t_te_labels[non_member_mask]))
            
            attack_X_test = np.vstack((member_X_test, non_member_X_test))
            attack_y_test = np.hstack((np.ones(len(member_X_test)), np.zeros(len(non_member_X_test))))
            
            attack_X_test = torch.tensor(attack_X_test, dtype=torch.float32)
            
            model_path = os.path.join(self.model_dir, f"attack_model_class_{class_idx}.pth")
            if not os.path.exists(model_path):
                print(f"Model for class {class_idx} not found. Skipping...")
                continue
            
            model = NNAttack(attack_X_test.shape[-1], self.hidden_size, 1)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                outputs = model(attack_X_test).cpu().numpy()
            
            y_pred = (outputs > 0.5).astype(int)
            all_preds.extend(y_pred.flatten())
            all_labels.extend(attack_y_test)
            all_preds_probs.extend(outputs.flatten())
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        if eval_metric == 'accuracy':
            return accuracy_score(all_labels, all_preds)
        elif eval_metric == 'auc':
            return roc_auc_score(all_labels, all_preds)
        elif eval_metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(all_labels, all_preds)
        elif eval_metric == 'TPR@0.01FPR':
            return self._compute_tpr_at_fpr(all_labels, all_preds_probs, fpr_threshold=0.01)
        elif eval_metric == 'TPR@0.001FPR':
            return self._compute_tpr_at_fpr(all_labels, all_preds_probs, fpr_threshold=0.001)
        elif eval_metric == 'TPR@0.0001FPR':
            return self._compute_tpr_at_fpr(all_labels, all_preds_probs, fpr_threshold=0.0001)
        else:
            raise ValueError(f"Unsupported evaluation metric: {eval_metric}")
    
    def _compute_tpr_at_fpr(self, y_true, y_scores, fpr_threshold=0.001):
        """计算在指定FPR下的TPR"""
        from sklearn.metrics import roc_curve
        
        if len(np.unique(y_scores)) < 2:
            return 0.0
            
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # 找到最接近1% FPR的索引
        closest_index = np.argmin(np.abs(fpr - fpr_threshold))
        tpr_at_fpr = tpr[closest_index]
        return tpr_at_fpr

    def infer(self, target_performance):

        self.t_outputs, self.t_labels = target_performance
        
        for class_idx in range(self.num_classes):
            mask = (self.t_labels == class_idx).reshape(-1, )
            attack_X = np.column_stack((self.t_outputs[mask], self.t_labels[mask]))

            attack_X = torch.tensor(attack_X, dtype=torch.float32)
            model_path = os.path.join(self.model_dir, f"attack_model_class_{class_idx}.pth")
            if not os.path.exists(model_path):
                print(f"Model for class {class_idx} not found. Skipping...")
                continue
            model = NNAttack(attack_X.shape[-1], self.hidden_size, 1)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                outputs = model(attack_X).cpu().numpy()

        return outputs


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import os
import time
import io
import pickle
import random 
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
except ImportError:
    from torch_geometric.nn.conv import GATConv
    from torch_geometric.nn.pool import global_mean_pool, global_max_pool
# 移除重复的rdkit导入
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
import traceback
from unimol_tools import MolTrain, MolPredict
import tempfile
import shutil
import sys
import subprocess

# 设置页面标题和布局 - 这必须是第一个 Streamlit 命令
st.set_page_config(page_title="2030 QSAR", layout="wide")

# ========== 修复1：定义page变量（Streamlit多页面/单页面兼容） ==========
# 方式1：单页面模式（直接指定page值）
page = "训练并推理深度学习模型"

# 方式2（可选）：如果是多页面应用，用sidebar选择页面（按需启用）
# page = st.sidebar.selectbox(
#     "选择功能页面",
#     ["训练并推理深度学习模型", "其他页面1", "其他页面2"]
# )

# ========== 页面逻辑 ==========
if page == "训练并推理深度学习模型":
    st.title("Train and Inference Deep Learning Model")
    
    # 初始化Session State
    if 'graph_df' not in st.session_state:
        st.session_state.graph_df = None
    if 'graph_model' not in st.session_state:
        st.session_state.graph_model = None
    if 'graph_scaler' not in st.session_state:
        st.session_state.graph_scaler = None

    # 1. 上传数据集
    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader("Select CSV file with `smiles` and `label` columns", type=["csv"], key="deep_learning_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {'smiles', 'label'}
            if not required_columns.issubset(df.columns):
                st.error(f"Missing required columns. Please ensure the file contains: {required_columns}")
            else:
                # 标签处理：确保为整数，过滤无效标签
                df['label'] = pd.to_numeric(df['label'], errors='coerce')
                df = df.dropna(subset=['label'])
                df['label'] = df['label'].astype(int)
                st.session_state.graph_df = df.copy()
                st.success("File uploaded successfully! Data Preview:")
                st.dataframe(df.head())
                
                # 显示类别分布
                st.subheader("Class Distribution")
                class_dist = df['label'].value_counts().sort_index()
                fig, ax = plt.subplots()
                class_dist.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Original Class Distribution')
                ax.set_xlabel('Class')
                ax.set_ylabel('Number of Samples')
                st.pyplot(fig)
                
                # 显示类别占比
                class_ratio = (class_dist / len(df) * 100).round(2)
                st.write("Class Ratio:")
                for cls, ratio in class_ratio.items():
                    st.write(f"Class {cls}: {ratio}% ({class_dist[cls]} samples)")
                
        except Exception as e:
            st.error(f"Failed to read file: {str(e)}")

    # 2. 数据预处理与平衡（简化版：手动过采样+欠采样）
    st.subheader("2. Data Preprocessing & Balancing")
    # ========== 修复2：将target_samples输入移到按钮外，避免按钮点击后输入框消失 ==========
    target_samples = st.number_input("Target samples per class", min_value=50, max_value=600, value=600, key="target_samples")
    
    if st.button("Preprocess Data and Create Molecular Graphs", key="preprocess_graph"):
        if st.session_state.graph_df is None:
            st.warning("Please upload dataset first")
        else:
            with st.spinner("Processing data..."):
                df = st.session_state.graph_df.copy()
                
                # 1. 定义分子图数据集（增强特征）
                class MoleculeDataset(Dataset):
                    def __init__(self, smiles_list, labels, transform=None):
                        super().__init__(transform)
                        self.smiles_list = smiles_list
                        self.labels = labels

                    def len(self):
                        return len(self.smiles_list)

                    def get(self, idx):
                        smiles = self.smiles_list[idx]
                        label = self.labels[idx]
                        mol = Chem.MolFromSmiles(smiles)
                        
                        if mol is None:
                            return None
                            
                        # 节点特征增强（8维）
                        x = []
                        for atom in mol.GetAtoms():
                            feats = [
                                atom.GetAtomicNum() / 10.0,  # Atomic number (normalized)
                                atom.GetFormalCharge(),
                                atom.GetHybridization().real,
                                atom.GetDegree() / 10.0,  # Degree (normalized)
                                atom.GetTotalNumHs() / 5.0,  # Number of H atoms (normalized)
                                1.0 if atom.GetIsAromatic() else 0.0,
                                atom.GetMass() / 50.0,  # Atomic mass (normalized)
                                1.0 if atom.GetSymbol() in ['O', 'N', 'S', 'P'] else 0.0  # Heteroatom indicator
                            ]
                            x.append(feats)
                        x = torch.tensor(x, dtype=torch.float)
                        
                        # 边特征增强（6维）
                        edge_index = []
                        edge_attr = []
                        for bond in mol.GetBonds():
                            u = bond.GetBeginAtomIdx()
                            v = bond.GetEndAtomIdx()
                            edge_index.append([u, v])
                            edge_index.append([v, u])  # Undirected graph
                            
                            # Edge features: explicit bond type encoding
                            bond_type = bond.GetBondType()
                            bond_feats = [
                                1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
                                1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
                                1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
                                1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
                                1.0 if bond.GetIsConjugated() else 0.0,
                                1.0 if bond.IsInRing() else 0.0
                            ]
                            edge_attr.append(bond_feats)
                            edge_attr.append(bond_feats)
                            
                        # Handle single-atom molecules with no bonds
                        if not edge_index:
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_attr = torch.empty((0, 6), dtype=torch.float)
                        else:
                            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                        
                        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))

                # 2. 创建数据集并过滤无效分子
                smiles_list = df['smiles'].tolist()
                labels = df['label'].tolist()
                dataset = MoleculeDataset(smiles_list, labels)
                valid_data = [data for data in dataset if data is not None]
                valid_labels = [data.y.item() for data in valid_data]
                st.success(f"Successfully created {len(valid_data)} molecular graphs (filtered out {len(dataset)-len(valid_data)} invalid molecules)")
                
                # 3. 手动数据平衡：过采样少数类 + 欠采样多数类（兼容所有环境）
                def balance_dataset(data_list, labels_list, target_samples=100):
                    """
                    Manual dataset balancing:
                    - Minority classes: Random oversampling to target sample count
                    - Majority classes: Random undersampling to target sample count
                    """
                    # Group data by class
                    class_groups = {}
                    for data, label in zip(data_list, labels_list):
                        if label not in class_groups:
                            class_groups[label] = []
                        class_groups[label].append(data)
                    
                    balanced_data = []
                    for cls, data in class_groups.items():
                        cls_count = len(data)
                        if cls_count < target_samples:
                            # Oversampling: Randomly duplicate samples
                            oversampled = data.copy()
                            while len(oversampled) < target_samples:
                                oversampled.append(random.choice(data))
                            balanced_data.extend(oversampled)
                            st.write(f"Class {cls}: {cls_count} samples → Oversampled to {target_samples} samples")
                        elif cls_count > target_samples:
                            # Undersampling: Randomly select samples
                            undersampled = random.sample(data, target_samples)
                            balanced_data.extend(undersampled)
                            st.write(f"Class {cls}: {cls_count} samples → Undersampled to {target_samples} samples")
                        else:
                            # Keep original count if matches target
                            balanced_data.extend(data)
                            st.write(f"Class {cls}: {cls_count} samples → Kept original count")
                    
                    # Shuffle balanced data
                    random.shuffle(balanced_data)
                    return balanced_data
                
                # 使用外部定义的target_samples（修复按钮点击后输入框消失问题）
                balanced_data = balance_dataset(valid_data, valid_labels, target_samples=target_samples)
                
                # 显示平衡结果
                balanced_labels = [data.y.item() for data in balanced_data]
                st.success(f"Dataset balancing completed: {len(valid_data)} original samples → {len(balanced_data)} balanced samples")
                
                # 可视化平衡后的分布
                fig, ax = plt.subplots()
                pd.Series(balanced_labels).value_counts().sort_index().plot(kind='bar', ax=ax, color='orange')
                ax.set_title('Balanced Class Distribution')
                ax.set_xlabel('Class')
                ax.set_ylabel('Number of Samples')
                st.pyplot(fig)
                
                # 4. 划分训练集和测试集
                if len(balanced_data) < 10:
                    st.error("Insufficient valid samples to split into train/test sets")
                else:
                    train_size = int(0.8 * len(balanced_data))
                    train_dataset = balanced_data[:train_size]
                    test_dataset = balanced_data[train_size:]
                    
                    # 训练集使用加权采样，进一步平衡训练过程
                    train_labels = [data.y.item() for data in train_dataset]
                    class_counts = pd.Series(train_labels).value_counts()
                    weights = 1.0 / class_counts
                    sample_weights = [weights[label] for label in train_labels]
                    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                    
                    st.session_state.train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
                    st.session_state.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                    st.session_state.full_dataset = balanced_data
                    
                    st.success(f"Train/test split completed: {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    # 3. 模型定义与训练
    st.subheader("3. Model Training")
    
    # 定义增强版GAT模型（残差连接+特征融合）
    class EnhancedGATModel(torch.nn.Module):
        def __init__(self, hidden_channels, num_node_features, num_classes, heads=4, dropout=0.3):
            super().__init__()
            torch.manual_seed(42)
            
            # Input projection (unify feature dimension)
            self.input_proj = torch.nn.Linear(num_node_features, hidden_channels)
            
            # GAT layers with residual connections
            self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, edge_dim=6, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=6, dropout=dropout)
            self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=6, dropout=dropout)
            
            # Residual projection layers (solve dimension mismatch)
            self.res_proj1 = torch.nn.Linear(hidden_channels, hidden_channels * heads)
            self.res_proj2 = torch.nn.Linear(hidden_channels * heads, hidden_channels * heads)
            
            # Classification head (two-layer fully connected)
            self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels // 2)  # *2 for merged pooling
            self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)
            
            # Batch normalization and Dropout
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
            self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn4 = torch.nn.BatchNorm1d(hidden_channels // 2)
            
            self.dropout = torch.nn.Dropout(dropout)
            self.relu = torch.nn.ELU()  # More stable than ReLU

        def forward(self, x, edge_index, edge_attr, batch):
            # Input projection + activation
            x = self.input_proj(x)
            x = self.relu(x)
            
            # First GAT layer + residual connection
            residual1 = self.res_proj1(x)
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x + residual1  # Residual connection: alleviate gradient vanishing
            
            # Second GAT layer + residual connection
            residual2 = self.res_proj2(x)
            x = self.conv2(x, edge_index, edge_attr)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = x + residual2
            
            # Third GAT layer
            x = self.conv3(x, edge_index, edge_attr)
            x = self.bn3(x)
            x = self.relu(x)
            
            # Global pooling: merge mean and max pooling (enhance feature expression)
            x_mean = global_mean_pool(x, batch)  # Mean pooling
            x_max = global_max_pool(x, batch)    # Max pooling
            x = torch.cat([x_mean, x_max], dim=1)  # Feature fusion
            
            # Classification head
            x = self.lin1(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.lin2(x)
            
            return x

    # 定义加权Focal Loss（专门解决数据不平衡）
    class WeightedFocalLoss(torch.nn.Module):
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha  # Class weights (higher for minority classes)
            self.gamma = gamma  # Focusing parameter (reduce weight of easy samples)
            self.reduction = reduction

        def forward(self, inputs, targets):
            # Cross entropy loss with class weights
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
            # Sample confidence (probability of correct prediction)
            pt = torch.exp(-ce_loss)
            # Focal Loss core: (1-pt)^gamma penalizes easy samples
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
            # Aggregate loss
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    # 训练参数设置（可视化界面）
    col1, col2 = st.columns(2)
    with col1:
        hidden_dim = st.slider("Hidden Dimension", 64, 256, 128, help="Model capacity: higher = stronger fitting ability")
        epochs = st.slider("Number of Epochs", 20, 200, 80, help="Training iterations: too few = underfitting, too many = overfitting")
        heads = st.slider("GAT Attention Heads", 2, 8, 4, help="Parallel attention heads: more = richer features")
    with col2:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f", help="Step size: too small = slow training, too large = instability")
        gamma = st.slider("Focal Loss Gamma", 1.0, 5.0, 2.0, help="Focusing parameter: higher = more attention to minority classes")
        dropout = st.slider("Dropout Probability", 0.1, 0.5, 0.3, help="Prevent overfitting by randomly dropping features")

    # 损失函数选择
    loss_type = st.radio("Select Loss Function", ["Class-Weighted Cross Entropy", "Weighted Focal Loss"], help="Both suit imbalanced data; Focal Loss is more effective")

    if st.button("Start Training Enhanced GAT Model", key="train_gat"):
        if 'train_loader' not in st.session_state:
            st.warning("Please complete data preprocessing first")
        else:
            with st.spinner("Training model..."):
                # 获取数据集信息
                full_dataset = st.session_state.full_dataset
                num_classes = len(torch.unique(torch.tensor([data.y.item() for data in full_dataset])))
                num_node_features = full_dataset[0].x.shape[1]
                st.info(f"Model Configuration: Feature Dimension {num_node_features} | Number of Classes {num_classes} | Attention Heads {heads}")
                
                # 初始化模型
                model = EnhancedGATModel(
                    hidden_channels=hidden_dim,
                    num_node_features=num_node_features,
                    num_classes=num_classes,
                    heads=heads,
                    dropout=dropout
                )
                
                # 计算类别权重（进一步平衡损失）
                train_labels = [data.y.item() for data in st.session_state.train_loader.dataset]
                class_counts = torch.bincount(torch.tensor(train_labels))
                class_weights = len(train_labels) / (num_classes * class_counts.float())  # Higher weights for minority classes
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                class_weights = class_weights.to(device)
                
                # 定义损失函数和优化器
                if loss_type == "Class-Weighted Cross Entropy":
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                else:
                    criterion = WeightedFocalLoss(alpha=class_weights, gamma=gamma)
                
                # 优化器：Adam + 权重衰减（防止过拟合）
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                # 学习率衰减：验证损失不下降时降低学习率
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
                
                # 设备配置（自动检测GPU/CPU）
                model.to(device)
                criterion.to(device)
                
                # 训练记录（用于后续可视化）
                train_losses = []
                test_losses = []
                train_f1s = []
                test_f1s = []
                train_kappas = []
                test_kappas = []
                
                # 进度条和状态显示
                progress_bar = st.progress(0)
                status_text = st.empty()

                for epoch in range(epochs):
                    # ---------------------- 训练阶段 ----------------------
                    model.train()
                    total_loss = 0
                    all_preds = []
                    all_labels = []
                    
                    for batch in st.session_state.train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()  # Clear gradients
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # Forward pass
                        loss = criterion(out, batch.y.squeeze())  # Calculate loss
                        loss.backward()  # Backward pass
                        optimizer.step()  # Update parameters
                        
                        # Record loss and predictions
                        total_loss += loss.item()
                        preds = out.argmax(dim=1)  # Select class with highest probability
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch.y.cpu().numpy())
                    
                    # Calculate training metrics
                    train_loss = total_loss / len(st.session_state.train_loader)
                    train_f1 = f1_score(all_labels, all_preds, average='weighted')  # Weighted F1 for imbalanced data
                    train_kappa = cohen_kappa_score(all_labels, all_preds)  # Kappa: robust to class distribution
                    
                    # ---------------------- 测试阶段 ----------------------
                    model.eval()
                    total_test_loss = 0
                    test_preds = []
                    test_labels = []
                    
                    with torch.no_grad():  # Disable gradient computation to save memory
                        for batch in st.session_state.test_loader:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                            loss = criterion(out, batch.y.squeeze())
                            total_test_loss += loss.item()
                            
                            preds = out.argmax(dim=1)
                            test_preds.extend(preds.cpu().numpy())
                            test_labels.extend(batch.y.cpu().numpy())
                    
                    # Calculate test metrics
                    test_loss = total_test_loss / len(st.session_state.test_loader)
                    test_f1 = f1_score(test_labels, test_preds, average='weighted')
                    test_kappa = cohen_kappa_score(test_labels, test_preds)
                    
                    # Learning rate scheduling (based on test loss)
                    scheduler.step(test_loss)
                    
                    # Record metrics
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    train_f1s.append(train_f1)
                    test_f1s.append(test_f1)
                    train_kappas.append(train_kappa)
                    test_kappas.append(test_kappa)
                    
                    # Update progress bar and status
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Train F1: {train_f1:.4f} | "
                        f"Test F1: {test_f1:.4f} | "
                        f"Kappa: {test_kappa:.4f}"
                    )
                
                # 保存模型和结果（CPU版本，避免设备不兼容）
                st.session_state.graph_model = model.cpu()
                st.session_state.test_labels = test_labels
                st.session_state.test_preds = test_preds
                st.session_state.num_classes = num_classes
                
                # 训练完成提示
                st.success("Enhanced GAT Model Training Completed!")
                
                # ---------------------- 训练结果可视化 ----------------------
                fig, axes = plt.subplots(1, 3, figsize=(21, 6))
                
                # 1. 损失曲线
                axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='#1f77b4')
                axes[0].plot(test_losses, label='Test Loss', linewidth=2, color='#ff7f0e')
                axes[0].set_title('Loss Curve', fontsize=12)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 2. 加权F1曲线
                axes[1].plot(train_f1s, label='Train Weighted F1', linewidth=2, color='#1f77b4')
                axes[1].plot(test_f1s, label='Test Weighted F1', linewidth=2, color='#ff7f0e')
                axes[1].set_title('Weighted F1 Score Curve', fontsize=12)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('F1 Score')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # 3. Kappa曲线
                axes[2].plot(train_kappas, label='Train Kappa', linewidth=2, color='#1f77b4')
                axes[2].plot(test_kappas, label='Test Kappa', linewidth=2, color='#ff7f0e')
                axes[2].set_title('Cohen Kappa Curve', fontsize=12)
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Kappa Score')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示最终性能指标
                st.subheader("Final Model Performance Metrics")
                final_metrics = pd.DataFrame({
                    'Metric': ['Weighted F1', 'Cohen Kappa', 'Test Loss'],
                    'Train Set': [f"{train_f1s[-1]:.4f}", f"{train_kappas[-1]:.4f}", f"{train_losses[-1]:.4f}"],
                    'Test Set': [f"{test_f1s[-1]:.4f}", f"{test_kappas[-1]:.4f}", f"{test_losses[-1]:.4f}"]
                })
                st.dataframe(final_metrics, use_container_width=True)
                
                # 显示混淆矩阵
                st.subheader("Test Set Confusion Matrix")
                cm = confusion_matrix(test_labels, test_preds)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(torch.unique(torch.tensor(test_labels)).numpy()),
                    yticklabels=sorted(torch.unique(torch.tensor(test_labels)).numpy()),
                    ax=ax
                )
                ax.set_xlabel('Predicted Class', fontsize=12)
                ax.set_ylabel('True Class', fontsize=12)
                ax.set_title('Confusion Matrix (Rows=True, Columns=Predicted)', fontsize=14)
                st.pyplot(fig)

    # 4. 模型推理（预测功能）
    st.subheader("4. Model Inference")
    predict_input = st.radio("Select Prediction Input Type", ["Manual SMILES Input", "Upload CSV File"], key="predict_input_type")
    
    # 输入处理
    predict_smiles = ""  # 初始化变量，避免未定义
    if predict_input == "Manual SMILES Input":
        predict_smiles = st.text_area("Enter SMILES for prediction (one per line)", value="CCO\nCC(=O)O\nC1=CC=CC=C1\nNCC(=O)O", help="Example: CCO (Ethanol), C1=CC=CC=C1 (Benzene)")
    else:
        predict_file = st.file_uploader("Upload CSV file with `smiles` column", type=["csv"], key="predict_file")

    if st.button("Start Prediction", key="predict_gat"):
        if st.session_state.graph_model is None:
            st.warning("Please train the model first")
        else:
            with st.spinner("Predicting..."):
                model = st.session_state.graph_model
                model.eval()
                smiles_list = []
                
                # 解析输入
                if predict_input == "Manual SMILES Input":
                    smiles_list = [s.strip() for s in predict_smiles.split('\n') if s.strip()]
                else:
                    if predict_file is not None:
                        pred_df = pd.read_csv(predict_file)
                        if 'smiles' not in pred_df.columns:
                            st.error("CSV file must contain `smiles` column")
                            st.stop()
                        smiles_list = pred_df['smiles'].tolist()
                    else:
                        st.error("Please upload CSV file")
                        st.stop()
                
                # 预测结果存储
                results = []
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                for smiles in smiles_list:
                    # 解析SMILES为分子
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results.append({
                            'SMILES': smiles,
                            'Predicted Class': 'Invalid Molecule',
                            'Confidence': 'N/A',
                            'Class Probabilities': 'N/A'
                        })
                        continue
                    
                    # 构建分子图（与训练时特征一致）
                    x = []
                    for atom in mol.GetAtoms():
                        feats = [
                            atom.GetAtomicNum() / 10.0,
                            atom.GetFormalCharge(),
                            atom.GetHybridization().real,
                            atom.GetDegree() / 10.0,
                            atom.GetTotalNumHs() / 5.0,
                            1.0 if atom.GetIsAromatic() else 0.0,
                            atom.GetMass() / 50.0,
                            1.0 if atom.GetSymbol() in ['O', 'N', 'S', 'P'] else 0.0
                        ]
                        x.append(feats)
                    x = torch.tensor(x, dtype=torch.float).to(device)
                    
                    # 构建边特征
                    edge_index = []
                    edge_attr = []
                    for bond in mol.GetBonds():
                        u = bond.GetBeginAtomIdx()
                        v = bond.GetEndAtomIdx()
                        edge_index.append([u, v])
                        edge_index.append([v, u])
                        
                        bond_type = bond.GetBondType()
                        bond_feats = [
                            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
                            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
                            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
                            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
                            1.0 if bond.GetIsConjugated() else 0.0,
                            1.0 if bond.IsInRing() else 0.0
                        ]
                        edge_attr.append(bond_feats)
                        edge_attr.append(bond_feats)
                    
                    # 处理无键分子
                    if not edge_index:
                        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
                        edge_attr = torch.empty((0, 6), dtype=torch.float).to(device)
                    else:
                        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)
                    
                    # 批处理标识（单样本batch=0）
                    batch = torch.zeros(x.shape[0], dtype=torch.long).to(device)
                    
                    # 预测
                    with torch.no_grad():
                        out = model(x, edge_index, edge_attr, batch)
                        probs = F.softmax(out, dim=1)  # Convert to probabilities
                        pred_label = out.argmax(dim=1).item()  # Predicted class
                        confidence = probs[0, pred_label].item()  # Prediction confidence
                        # Format class probabilities
                        prob_str = ", ".join([f"Class {i}: {p:.3f}" for i, p in enumerate(probs[0].cpu().numpy())])
                    
                    # 存储结果
                    results.append({
                        'SMILES': smiles,
                        'Predicted Class': pred_label,
                        'Confidence': f"{confidence:.4f}",
                        'Class Probabilities': prob_str
                    })
                
                # 显示预测结果
                st.subheader("Prediction Results")
                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True)
                
                # 下载结果
                csv_data = result_df.to_csv(index=False, encoding='utf-8').encode('utf-8')
                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="pyg_gat_predictions.csv",
                    mime="text/csv",
                    key='download-predictions'
                )
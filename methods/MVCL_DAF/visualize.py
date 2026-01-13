# visualize.py

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 需要从您的项目中导入相关模块
from methods import method_map
from data.base import DataManager
from configs.base import ParamManager, add_config_param
from utils.functions import set_torch_seed
from utils.functions import set_output_path

def visualize_causal_invariance(args):
    # --- 初始化模型和数据加载器 ---
    # 这部分代码与 run.py 类似
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    param = ParamManager(args)
    args = param.args
    args = add_config_param(args, args.config_file_name)
    save_model_name = f"{args.method}_{args.dataset}_{args.text_backbone}_{args.data_mode}_{args.seed}"
    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)
    set_torch_seed(args.seed)
    
    data = DataManager(args)
    labels_weight = data.labels_weight
    
    method_manager_class = method_map[args.method]
    method_manager = method_manager_class(args, data, labels_weight=labels_weight)
    
    # --- 步骤 1: 提取特征 ---
    print("Extracting features for t-SNE visualization...")
    features, labels, sources = method_manager._extract_features_for_tsne(args)
    
    if features is None:
        print("Feature extraction failed. Exiting.")
        return

    print(f"Extracted {features.shape[0]} feature vectors.")

    # --- 步骤 2: 执行 t-SNE 降维 ---
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=args.seed)
    
    tsne_results = tsne.fit_transform(features)
    
    # --- 步骤 3: 创建 DataFrame 并进行可视化 ---
    print("Creating visualization...")
    df = pd.DataFrame({
        'tsne-2d-one': tsne_results[:,0],
        'tsne-2d-two': tsne_results[:,1],
        'intent_label': labels,
        'feature_source': sources
    })
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", 
        y="tsne-2d-two",
        hue="intent_label",     # 用颜色区分意图
        style="feature_source", # 用形状区分特征来源
        palette=sns.color_palette("hls", n_colors=len(np.unique(labels))),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Intent Representation Space')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 将图例放在图外
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # 保存图像
    output_filename = f"tsne_causal_invariance_{args.dataset}_{args.seed}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_filename}")


if __name__ == '__main__':
    # 从 run.py 复制 argparse 部分，并作少量修改
    parser = argparse.ArgumentParser()
    # 保持与 run.py 一致的参数
    parser.add_argument('--dataset', type=str, default='MIntRec2', help="The selected dataset.")
    parser.add_argument('--method', type=str, default='mvcl_daf', help="Which method to use.")
    parser.add_argument('--config_file_name', type=str, default='MVCL_DAF_MIntRec.py')
    parser.add_argument('--seed', type=int, default=2, help="Random seed.")
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU ID.")
    # ... 您可以从 run.py 复制所有必要的参数 ...
    parser.add_argument('--logger_name', type=str, default='Multimodal Intent Recognition', help="Logger name for multimodal intent recognition.")
    parser.add_argument('--data_mode', type=str, default='multi-class', help="The selected person id.")
    parser.add_argument("--text_backbone", type=str, default='bert-large-uncased', help="which backbone to use for text modality")
    parser.add_argument('--num_workers', type=int, default=8, help="The number of workers to load data.")
    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")
    parser.add_argument("--data_path", default="/root/autod_l-tmp/MVCL-DAF/dataset/", type=str) # 请确保路径正确
    parser.add_argument("--train", action="store_true", default=False) # 在这里设为False
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_results", action="store_true", default=False)
    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")
    parser.add_argument('--cache_path', type=str, default='/root/autod-tmp/model/bert-large-uncased/') # 请确保路径正确
    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")
    parser.add_argument("--output_path", default= 'outputs', type=str)
    parser.add_argument("--model_path", default= 'models', type=str)

    args = parser.parse_args()
    
    visualize_causal_invariance(args)


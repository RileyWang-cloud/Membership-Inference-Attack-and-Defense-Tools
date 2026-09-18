
import copy
import sys
import os
import numpy as np
import random
import tqdm
import argparse
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from mia_evals.dataset_utils import load_member_data
from absl import flags
from model import UNet
import torch
import torch.nn as nn
from mia_evals.resnet import ResNet18

# font_path = './SIMSUNB.TTF'
# font_prop = font_manager.FontProperties(fname=font_path)
# matplotlib.rcParams['font.family'] = font_prop.get_name()
# 避免负号显示为方块
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#######################
# Wrapper for ensemble
#######################
class EnsembleUNetWrapper(nn.Module):
    def __init__(self, models_list, t_to_group):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.t_to_group = torch.tensor(t_to_group).long()

    def forward(self, x, t):
        group_ids = self.t_to_group[t.cpu()].to(x.device)  # (batch,)
        out = torch.zeros_like(x)
        for g in group_ids.unique():
            idx = (group_ids == g).nonzero(as_tuple=False).squeeze()
            if idx.ndim == 0:
                idx = idx.unsqueeze(0)
            out[idx] = self.models[g](x[idx], t[idx])
        return out

######################
# t_to_group util
######################
def generate_t_to_group(T, num_t_groups):
    t_to_group = []
    step = T // num_t_groups
    for g in range(num_t_groups):
        t_to_group += [g] * step
    t_to_group += [num_t_groups - 1] * (T - len(t_to_group))
    return t_to_group


def infer_grouped_config_from_ckpt(state_dict, FLAGS):
    keys = list(state_dict.keys())
    has_module_prefix = any(k.startswith("module.") for k in keys)

    def strip_module_prefix(key):
        return key[len("module."):] if key.startswith("module.") else key

    stripped_keys = [strip_module_prefix(k) for k in keys]

    k = FLAGS.k
    T = FLAGS.T

    if "module.M" in state_dict:
        M_tensor = state_dict["module.M"]
        k = int(M_tensor.shape[0])
        T = int(M_tensor.shape[1])
    elif "M" in state_dict:
        M_tensor = state_dict["M"]
        k = int(M_tensor.shape[0])
        T = int(M_tensor.shape[1])

    model_indices = set()
    for key in stripped_keys:
        if key.startswith("models."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                model_indices.add(int(parts[1]))

    if len(model_indices) > 0:
        num_t_groups = max(model_indices) + 1
    else:
        num_t_groups = FLAGS.num_t_groups

    if "module.t_to_group" in state_dict:
        t_to_group = state_dict["module.t_to_group"].cpu().tolist()
    elif "t_to_group" in state_dict:
        t_to_group = state_dict["t_to_group"].cpu().tolist()
    else:
        t_to_group = generate_t_to_group(T, num_t_groups)

    return {
        "has_module_prefix": has_module_prefix,
        "k": k,
        "T": T,
        "num_t_groups": num_t_groups,
        "t_to_group": t_to_group,
    }


def build_and_load_grouped_trainer(ckpt_path, FLAGS, trainer_cls, trainer_kwargs_extra=None):
    if trainer_kwargs_extra is None:
        trainer_kwargs_extra = {}

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["trainer"]
    inferred = infer_grouped_config_from_ckpt(state_dict, FLAGS)

    T = inferred["T"]
    k = inferred["k"]
    num_t_groups = inferred["num_t_groups"]
    t_to_group = inferred["t_to_group"]
    has_module_prefix = inferred["has_module_prefix"]

    print(
        f"[Auto-infer] T={T}, k={k}, num_t_groups={num_t_groups}, "
        f"has_module_prefix={has_module_prefix}"
    )

    trainer = trainer_cls(
        model_class=lambda: UNet(
            T=T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult,
            attn=FLAGS.attn, num_res_blocks=FLAGS.num_res_blocks,
            dropout=FLAGS.dropout),
        beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T, T=T, k=k,
        num_t_groups=num_t_groups, t_to_group=t_to_group,
        **trainer_kwargs_extra).to(device)

    if has_module_prefix:
        trainer = torch.nn.DataParallel(trainer)

    trainer.load_state_dict(state_dict)
    return trainer, inferred


def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device=device):

    x = x.to(device)

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)

    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    if requires_grad:
        epsilon = model(x, t_c)
    else:
        with torch.no_grad():
            epsilon = model(x, t_c)

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                 + (1 - alphas_t_target).sqrt() * epsilon

    return {
        'x_t_target': x_t_target,
        'epsilon': epsilon
    }


def ddim_multistep(model, FLAGS, x, t_c, target_steps, clip=False, device=device, requires_grad=False):
    # eg: target_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90], t_c=0
    for idx, t_target in enumerate(target_steps):
        result = ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device)
        x = result['x_t_target']
        t_c = t_target

    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

    return result


class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_FLAGS(flag_path):
    FLAGS = flags.FLAGS
    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


def build_attack_flags(flag_path, args, ckpt_name):
    FLAGS = get_FLAGS(flag_path)
    FLAGS(["secmia.py"])

    FLAGS.ckpt_name = ckpt_name
    FLAGS.output_save_dir = args.output_save_dir
    FLAGS.dataset = args.dataset
    FLAGS.data_root = args.data_root
    FLAGS.t_sec = args.t_sec
    FLAGS.m = args.m
    FLAGS.model_type = args.model_type
    FLAGS.k = args.k
    FLAGS.num_t_groups = args.num_t_groups

    return FLAGS

def define_flags():
    flags.DEFINE_bool('train', False, help='train from scratch')
    flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
    # UNet
    flags.DEFINE_integer('ch', 128, help='base channel of UNet')
    flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
    flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
    flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
    flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
    # Gaussian Diffusion
    flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
    flags.DEFINE_float('beta_T', 0.02, help='end beta value')
    flags.DEFINE_integer('T', 1000, help='total diffusion steps')  # 时间步
    flags.DEFINE_integer('k', 10, help='total training groups')  # 训练组数量
    flags.DEFINE_integer('num_t_groups', 1, help='total denoising networks')  # 独立去噪网络的数量
    flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
    flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
    # Training
    flags.DEFINE_float('lr', 2e-4, help='target learning rate')
    flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
    flags.DEFINE_integer('total_steps', 20, help='total training steps')
    flags.DEFINE_integer('img_size', 32, help='image size')
    flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
    flags.DEFINE_integer('batch_size', 128, help='batch size')
    flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
    flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
    flags.DEFINE_bool('parallel', True, help='multi gpu training')
    
    
    flags.DEFINE_bool('use_predefined_M', False, help='Use predefined encoding matrix M')
    flags.DEFINE_string('M_path', './logs/ETDM_CIFAR10/M.pt', help='Path to load predefined M')  # 预定义编码矩阵M

    # Logging & Sampling
    flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, "sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 10, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

    flags.DEFINE_string('model_dir', './pretrained_DDPM', help='Path to target model')
    flags.DEFINE_string('data_root', '', help='Path to dataset')
    flags.DEFINE_string('dataset', 'cifar10', help='Type of dataset')
    flags.DEFINE_string('device', 'cuda', help='Device')
    flags.DEFINE_integer('m', 10, help='DDIM interval')
    flags.DEFINE_integer('t_sec', 100, help='timestep used for error comparing')
    flags.DEFINE_float('sparsity', '0.3', help='Sparsity of mask matrix')
    flags.DEFINE_string('model_type', 'ddpm', help='Type of the target model')
    flags.DEFINE_string('ckpt_name', 'ckpt-step8000', help='Name of the model file')
    flags.DEFINE_list('ckpt_names', [], help='List of ckpt names to evaluate in batch mode')
    # flags.DEFINE_list('model_dirs', [], help='List of model names to evaluate in batch mode')
    flags.DEFINE_string('output_save_dir', './experiment_results/recent',help='Dir to save attack results')



def get_model(ckpt, FLAGS, WA=True):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    print(ckpt)
    ckpt = torch.load(ckpt, map_location=torch.device(device))

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('_module.'):
            new_state_dict.update({key[8:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


def get_model_selective(ckpt_path, FLAGS):
    """Load an ADF/SMCD checkpoint as a regular ``model(images, timesteps)`` denoiser."""
    from Defense.adf_diffusion import SelectiveDenoiser
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    state = checkpoint["ema_model"] if "ema_model" in checkpoint else checkpoint["trainer"]
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    mask = state.get("M")
    if mask is None:
        raise ValueError("Invalid SMCD checkpoint: missing trainer mask M.")
    t_to_group = state.get("t_to_group")
    if t_to_group is None:
        t_to_group = torch.tensor(generate_t_to_group(int(mask.shape[1]), int(FLAGS.num_t_groups)))
    group_count = max(int(key.split(".")[1]) for key in state if key.startswith("models.")) + 1
    models = [UNet(T=int(mask.shape[1]), ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                   num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout) for _ in range(group_count)]
    model = SelectiveDenoiser(models, t_to_group.tolist(), mask)
    model.load_state_dict({key: value for key, value in state.items() if key in model.state_dict()}, strict=False)
    return model.eval()


def calculate_fid(model, FLAGS, num_images=100, batch_size=None, fid_cache=None,
                  use_torch=False, verbose=True, device=None):
    """Generate DDPM samples from ``model`` and return their FID.

    The statistics cache defaults to this utility's CIFAR-10 training cache.
    ``num_images`` should be large (for example 10,000+) for a final result.
    """
    try:
        from .fid_evaluation import calculate_model_fid
    except ImportError:  # Support invoking this legacy utility as a script.
        from fid_evaluation import calculate_model_fid
    return calculate_model_fid(
        model, FLAGS, num_images=num_images, batch_size=batch_size,
        fid_cache=fid_cache, use_torch=use_torch, verbose=verbose, device=device,
    )


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def norm(x):
    return (x + 1) / 2


def get_intermediate_results(model, FLAGS, data_loader, t_sec, timestep):

    target_steps = list(range(0, t_sec, timestep))[1:]  # eg：t_sec=100，timestep=10，target_steps=[10, 20, 30, 40, 50, 60, 70, 80, 90]

    internal_diffusion_list = []
    internal_denoised_list = []
    for batch_idx, x in enumerate(tqdm.tqdm(data_loader)):
        x = x[0].to(device)
        # x = x[0].cpu()
        x = x * 2 - 1

        x_sec = ddim_multistep(model, FLAGS, x, t_c=0, target_steps=target_steps)
        x_sec = x_sec['x_t_target']  # 前向加噪到target_steps[-1]，得到x_sec
        x_sec_recon = ddim_singlestep(model, FLAGS, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)  # 进一步从target_steps[-1]再加噪timestep步，eg：90->100
        x_sec_recon = ddim_singlestep(model, FLAGS, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep, t_target=target_steps[-1])  # 去噪回target_steps[-1]，eg：100->90
        x_sec_recon = x_sec_recon['x_t_target']  # 反向去噪得到的x_tsec_recon

        internal_diffusion_list.append(x_sec)  # 扩散的中间结果
        internal_denoised_list.append(x_sec_recon)  # 去噪的中间结果

    return {
        'internal_diffusions': torch.concat(internal_diffusion_list),
        'internal_denoise': torch.concat(internal_denoised_list)
    }


def save_roc_curve(FLAGS, fpr, tpr, auc_val, ckpt_name, save_dir, prefix='SecMI'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{prefix}_{ckpt_name}_{FLAGS.t_sec}.png')

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'[✓] ROC curve saved to: {save_path}')

def save_combined_roc_curve(FLAGS, stat_fpr, stat_tpr, stat_auc,
                             nns_fpr, nns_tpr, nns_auc,
                             ckpt_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'SecMI_Combined_{ckpt_name}_{FLAGS.t_sec}.png')

    plt.figure()
    plt.plot(stat_fpr, stat_tpr, label=f'Stat_AUC = {stat_auc:.4f}')
    plt.plot(nns_fpr, nns_tpr, label=f'NNs_AUC = {nns_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SecMI ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'[✓] Combined ROC curve saved to: {save_path}')


def save_attack_results(result_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(result_dict, save_path)
    print(f"[✓] Attack result saved to: {save_path}")


def calculate_auc_asr_stat(member_scores, nonmember_scores):
    print(f'member score: {member_scores.mean():.4f} nonmember score: {nonmember_scores.mean():.4f}')

    total = member_scores.size(0) + nonmember_scores.size(0)

    min_score = min(member_scores.min(), nonmember_scores.min()).item()
    max_score = min(member_scores.max(), nonmember_scores.max()).item()
    print(min_score, max_score)

    TPR_list = []
    FPR_list = []

    best_asr = 0

    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 1000):
        acc = ((member_scores >= threshold).sum() + (nonmember_scores < threshold).sum()) / total

        TP = (member_scores >= threshold).sum()
        TN = (nonmember_scores < threshold).sum()
        FP = (nonmember_scores >= threshold).sum()
        FN = (member_scores < threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        ASR = (TP + TN) / (TP + TN + FP + FN)

        if ASR > best_asr:
            best_asr = ASR

        if FPR_1_idx > (0.01 - FPR).abs():
            FPR_1_idx = (0.01 - FPR).abs()
            TPRatFPR_1 = TPR

        if FPR_01_idx > (0.001 - FPR).abs():
            FPR_01_idx = (0.001 - FPR).abs()
            TPRatFPR_01 = TPR

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.4f} \t TPR: {TPR:.4f} \t FPR: {FPR:.4f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc} \t ASR: {best_asr} \t TPR@FPR=1%: {TPRatFPR_1} \t TPR@FPR=0.1%: {TPRatFPR_01}')


def secmi_attack(model, FLAGS, dataset_root, timestep=10, t_sec=100, batch_size=128, dataset='cifar10'):
    ckpt_name = FLAGS.ckpt_name.split('.')[0]

    stat_save_path = os.path.join(FLAGS.output_save_dir, 'SecMI_attack_results', f'SecMI_Stat_{ckpt_name}_{FLAGS.t_sec}.pt')
    print(stat_save_path)
    nns_save_path = os.path.join(FLAGS.output_save_dir, 'SecMI_attack_results', f'SecMI_NNs_{ckpt_name}_{FLAGS.t_sec}.pt')
    print(nns_save_path)
    save_dir=os.path.join(FLAGS.output_save_dir, 'SecMI_roc')
    print(save_dir)
    # load splits
    print(dataset_root)
    _, _, member_loader, nonmember_loader = load_member_data(dataset_root=dataset_root, dataset_name=dataset, batch_size=batch_size,
                                                             shuffle=False, randaugment=False)

    member_results = get_intermediate_results(model, FLAGS, member_loader, t_sec, timestep)
    nonmember_results = get_intermediate_results(model, FLAGS, nonmember_loader, t_sec, timestep)
    # print(member_results)
    # print(nonmember_results)

    ckpt_name = FLAGS.ckpt_name.split('.')[0]

    t_results = {
        'member_diffusions': member_results['internal_diffusions'],
        'member_internal_samples': member_results['internal_denoise'],
        'nonmember_diffusions': nonmember_results['internal_diffusions'],
        'nonmember_internal_samples': nonmember_results['internal_denoise'],
    }
    # print(t_results)
    
    stat_results = execute_attack(t_results, type='stat')
    print('#' * 20 + ' SecMI_stat ' + '#' * 20)
    print_result(stat_results)
    nns_results = execute_attack(t_results, type='nns')
    print('#' * 20 + ' SecMI_NNs ' + '#' * 20)
    print_result(nns_results)

    # 保存 stat 结果
    stat_save_path = os.path.join(FLAGS.output_save_dir, 'SecMI_attack_results', f'SecMI_Stat_{ckpt_name}_{FLAGS.t_sec}.pt')
    save_attack_results(stat_results, stat_save_path)

    # 保存 nns 结果
    nns_save_path = os.path.join(FLAGS.output_save_dir, 'SecMI_attack_results', f'SecMI_NNs_{ckpt_name}_{FLAGS.t_sec}.pt')
    save_attack_results(nns_results, nns_save_path)

    # 保存 ROC 曲线图像
    save_roc_curve(FLAGS, stat_results['fpr_list'].numpy(), stat_results['tpr_list'].numpy(),
                   stat_results['auc'], ckpt_name=ckpt_name, save_dir=save_dir, prefix='SecMI_Stat')

    save_roc_curve(FLAGS, nns_results['fpr_list'].numpy(), nns_results['tpr_list'].numpy(),
                   nns_results['auc'], ckpt_name=ckpt_name, save_dir=save_dir, prefix='SecMI_NNs')
    # 保存组合 ROC 曲线
    save_combined_roc_curve(
        FLAGS,
        stat_fpr=stat_results['fpr_list'].numpy(),
        stat_tpr=stat_results['tpr_list'].numpy(),
        stat_auc=stat_results['auc'],
        nns_fpr=nns_results['fpr_list'].numpy(),
        nns_tpr=nns_results['tpr_list'].numpy(),
        nns_auc=nns_results['auc'],
        ckpt_name=ckpt_name,
        save_dir=save_dir
    )

def print_result(results):
    keys = ['auc', 'asr', 'TPR@1%FPR', 'TPR@0.1%FPR', 'threshold']
    for k, v in results.items():
        if k in keys:
            print(f'{k}: {v}')

def naive_statistic_attack(t_results, metric='l2'):
    # 计算重建误差
    def measure(diffusion, sample, metric, device=device):
        diffusion = diffusion.to(device).float()
        sample = sample.to(device).float()

        if len(diffusion.shape) == 5:
            num_timestep = diffusion.size(0)
            diffusion = diffusion.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)
            sample = sample.permute(1, 0, 2, 3, 4).reshape(-1, num_timestep * 3, 32, 32)

        if metric == 'l2':
            score = ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
        else:
            raise NotImplementedError

        return score

    # member scores
    member_scores = measure(t_results['member_diffusions'], t_results['member_internal_samples'], metric=metric)
    # nonmember scores
    nonmember_scores = measure(t_results['nonmember_diffusions'], t_results['nonmember_internal_samples'],
                               metric=metric)
    return member_scores, nonmember_scores


def execute_attack(t_result, type):
    if type == 'stat':
        member_scores, nonmember_scores = naive_statistic_attack(t_result, metric='l2')
    elif type == 'nns':
        member_scores, nonmember_scores, model = nns_attack(t_result, train_portion=0.2, device=device)
        # 让 NNS attack 输出的分数和 naive_statistic_attack 的分数方向保持一致
        member_scores *= -1
        nonmember_scores *= -1
    else:
        raise NotImplementedError

    auc, asr, fpr_list, tpr_list, threshold = roc(member_scores, nonmember_scores, n_points=2000)
    # TPR @ 1% FPR
    tpr_1_fpr = tpr_list[(fpr_list - 0.01).abs().argmin(dim=0)]
    # TPR @ 0.1% FPR
    tpr_01_fpr = tpr_list[(fpr_list - 0.001).abs().argmin(dim=0)]

    exp_data = {
        'member_scores': member_scores,  # for histogram
        'nonmember_scores': nonmember_scores,
        'asr': asr.item(),
        'auc': auc,
        'fpr_list': fpr_list,
        'tpr_list': tpr_list,
        'TPR@1%FPR': tpr_1_fpr,
        'TPR@0.1%FPR': tpr_01_fpr,
        'threshold': threshold
    }

    return exp_data


def roc(member_scores, nonmember_scores, n_points=1000):
    max_asr = 0
    max_threshold = 0

    # min_conf ~ max_conf 范围内，遍历 1000 个阈值，逐步评估每个阈值的性能。
    min_conf = min(member_scores.min(), nonmember_scores.min()).item()
    max_conf = max(member_scores.max(), nonmember_scores.max()).item()

    FPR_list = []
    TPR_list = []

    for threshold in torch.arange(min_conf, max_conf, (max_conf - min_conf) / n_points):
        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)  # 真阳性率
        FPR = FP / (FP + TN)  # 假阳性率

        ASR = (TP + TN) / (TP + TN + FP + FN)  # 成员推理准确率

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())
        
        # 记录最优的asr和对应的阈值
        if ASR > max_asr:
            max_asr = ASR
            max_threshold = threshold

    FPR_list = np.asarray(FPR_list)
    TPR_list = np.asarray(TPR_list)
    FPR_list, unique_idx = np.unique(FPR_list, return_index=True)
    TPR_list = TPR_list[unique_idx]
    auc = metrics.auc(FPR_list, TPR_list)  # 衡量整体分类能力
    return auc, max_asr, torch.from_numpy(FPR_list), torch.from_numpy(TPR_list), max_threshold


def nns_attack(t_results, train_portion=0.5, device='cuda'):
    n_epoch = 15
    lr = 0.001
    batch_size = 128
    # model training
    train_loader, test_loader, num_timestep = split_nn_datasets(t_results, train_portion=train_portion,
                                                                batch_size=batch_size)
    print(f'num timestep: {num_timestep}')
    # initialize NNs
    model = ResNet18(num_channels=3 * num_timestep * 1, num_classes=1).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # model eval

    test_acc_best_ckpt = None
    test_acc_best = 0
    for epoch in range(n_epoch):
        train_loss, train_acc = nn_train(epoch, model, optim, train_loader)
        test_loss, test_acc = nn_eval(model, test_loader)
        if test_acc > test_acc_best:
            test_acc_best_ckpt = copy.deepcopy(model.state_dict())

    # resume best ckpt
    model.load_state_dict(test_acc_best_ckpt)
    model.eval()
    # generate member_scores, nonmember_scores
    member_scores = []
    nonmember_scores = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            logits = model(data.to(device))
            member_scores.append(logits[label == 1])
            nonmember_scores.append(logits[label == 0])

    member_scores = torch.concat(member_scores).reshape(-1)  # 攻击模型对member的预测得分，类似于置信度
    nonmember_scores = torch.concat(nonmember_scores).reshape(-1)  # 攻击模型对nonmember的预测得分，类似于置信度
    return member_scores, nonmember_scores, model


def nn_train(epoch, model, optimizer, data_loader, device=device):
    model.train()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device).reshape(-1, 1)

        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0
        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Epoch: {epoch} \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


def split_nn_datasets(t_results, train_portion=0.1, batch_size=128):
    # split training and testing
    # [t, 25000, 3, 32, 32]
    member_diffusion = t_results['member_diffusions']
    member_sample = t_results['member_internal_samples']
    nonmember_diffusion = t_results['nonmember_diffusions']
    nonmember_sample = t_results['nonmember_internal_samples']
    if len(member_diffusion.shape) == 4:
        # with one timestep
        # minus
        num_timestep = 1
        member_concat = (member_diffusion - member_sample).abs() ** 1
        nonmember_concat = (nonmember_diffusion - nonmember_sample).abs() ** 1
    elif len(member_diffusion.shape) == 5:
        # with multiple timestep
        # minus
        num_timestep = member_diffusion.size(0)
        member_concat = ((member_diffusion - member_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                       num_timestep * 3,
                                                                                                       32, 32)
        nonmember_concat = ((nonmember_diffusion - nonmember_sample).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                                                num_timestep * 3,
                                                                                                                32, 32)
    else:
        raise NotImplementedError

    # train num
    num_train = int(member_concat.size(0) * train_portion)
    # split
    train_member_concat = member_concat[:num_train]
    train_member_label = torch.ones(train_member_concat.size(0))  # 成员标签：1
    train_nonmember_concat = nonmember_concat[:num_train]
    train_nonmember_label = torch.zeros(train_nonmember_concat.size(0))  # 非成员标签：0
    test_member_concat = member_concat[num_train:]
    test_member_label = torch.ones(test_member_concat.size(0))
    test_nonmember_concat = nonmember_concat[num_train:]
    test_nonmember_label = torch.zeros(test_nonmember_concat.size(0))

    # datasets
    if num_train == 0:
        train_dataset = None
        train_loader = None
    else:
        train_dataset = MIDataset(train_member_concat, train_nonmember_concat, train_member_label,
                                  train_nonmember_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MIDataset(test_member_concat, test_nonmember_concat, test_member_label, test_nonmember_label)
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_timestep


@torch.no_grad()
def nn_eval(model, data_loader, device=device):
    model.eval()

    mean_loss = 0
    total = 0
    acc = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data, label = data.to(device), label.to(device).reshape(-1, 1)
        logit = model(data)

        loss = ((logit - label) ** 2).mean()

        mean_loss += loss.item()
        total += data.size(0)

        logit[logit >= 0.5] = 1
        logit[logit < 0.5] = 0

        acc += (logit == label).sum()

    mean_loss /= len(data_loader)
    print(f'Test: \t Loss: {mean_loss:.4f} \t Acc: {acc / total:.4f} \t')
    return mean_loss, acc / total


def batch_secmi_attack(args, ckpt_list, model_dir, data_root, dataset='cifar10', 
                       t_sec=100, timestep=10, model_type='ddpm', batch_size=1024):
    for ckpt_name in ckpt_list:
        print('=' * 60)
        print(f'[✓] 正在评估模型：{ckpt_name}')
        print('=' * 60)

        # 加载 FLAGS
        ckpt_path = os.path.join(model_dir, ckpt_name)
        flag_path = os.path.join(model_dir, 'flagfile.txt')
        FLAGS = build_attack_flags(flag_path, args, ckpt_name)

        # 载入模型
        if model_type == 'ddpm':
            model = get_model(ckpt_path, FLAGS, WA=True).to(device)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Only 'ddpm' and 'smcd' are supported.")

        # 更新 ckpt_name 进 FLAGS（用于文件命名）
        FLAGS.ckpt_name = ckpt_name

        print("当前保存目录为：", FLAGS.output_save_dir)
        # 执行攻击
        secmi_attack(model, FLAGS, dataset_root=data_root, t_sec=t_sec, timestep=timestep, batch_size=batch_size, dataset=dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./logs/ETDM_CIFAR10')
    # parser.add_argument('--model_dirs', nargs='+', default=['./logs/DDPM_CIFAR10', './logs/ETDM_CIFAR10'], help='List of model directories')
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--t_sec', type=int, default=100)
    parser.add_argument('--m', type=int, default=10)

    parser.add_argument('--k', type=int, default=3)  # 训练组数量
    parser.add_argument('--num_t_groups', type=int, default=5)

    parser.add_argument('--model_type', type=str, default='ddpm')
    # parser.add_argument('--ckpt_name', type=str, default='ckpt-step8000.pt')
    parser.add_argument('--ckpt_names', nargs='+', default=['ckpt-step8000.pt'], help='List of ckpt names to evaluate')
    parser.add_argument('--output_save_dir', type=str, default='./experiment_results/recent')
    
    args = parser.parse_args()

    fix_seed(0)
    
    define_flags()

    batch_secmi_attack(
        args=args,
        ckpt_list=args.ckpt_names,
        model_dir=args.model_dir,
        data_root=args.data_root,
        dataset=args.dataset,
        t_sec=args.t_sec,
        timestep=args.m,
        model_type=args.model_type,
        batch_size=1024
    )


    # ckpt = os.path.join(args.model_dir, args.ckpt_name)
    # flag_path = os.path.join(args.model_dir, 'flagfile.txt')
    # FLAGS = get_FLAGS(flag_path)
    # FLAGS(sys.argv)
    # # 判断模型类型
    # if args.model_type=='ddpm':
    #     # Load DDPM
    #     model = get_model(ckpt, FLAGS, WA=True).to(device)
    # elif args.model_type=='etdm':
    #     # Load ensemble model
    #     model = get_model_grouped(ckpt, FLAGS).to(device)
    # secmi_attack(model, FLAGS, dataset_root=args.data_root, t_sec=args.t_sec, timestep=args.m, batch_size=1024, dataset=args.dataset)

                                                                                      

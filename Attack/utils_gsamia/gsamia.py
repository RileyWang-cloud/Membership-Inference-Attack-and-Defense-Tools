
import copy
import sys
import os
import numpy as np
import random
import tqdm
import argparse

from sklearn import metrics
from mia_evals.dataset_utils import load_member_data
from absl import flags
from model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from mia_evals.resnet import ResNet18

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
    flags.DEFINE_integer('k', 3, help='total training groups')  # 训练组数量
    flags.DEFINE_integer('num_t_groups', 5, help='total denoising networks')  # 独立去噪网络的数量
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

    #gsa
    flags.DEFINE_string('output_name', '', help="The directory that save the gradient information.")
    flags.DEFINE_integer('sampling_frequency', 10, help="sampling frequency")
    flags.DEFINE_integer('attack_method', 1, help="GSA attack method: 1 or 2")
    flags.DEFINE_enum('prediction_type', 'epsilon', ['epsilon', 'sample'], help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",)

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


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
    """Generate DDPM samples from ``model`` and return their FID."""
    try:
        from .fid_evaluation import calculate_model_fid
    except ImportError:  # Support invoking this legacy utility as a script.
        from fid_evaluation import calculate_model_fid
    return calculate_model_fid(
        model, FLAGS, num_images=num_images, batch_size=batch_size,
        fid_cache=fid_cache, use_torch=use_torch, verbose=verbose, device=device,
    )

def get_dataset(dataset_root, dataset, batch_size):
    # load splits
    print(dataset_root)
    _, _, member_loader, nonmember_loader = load_member_data(dataset_root=dataset_root, dataset_name=dataset, batch_size=batch_size,
                                                             shuffle=False, randaugment=False)

    return member_loader, nonmember_loader


def extract_attack_features(
    model,
    model_name,
    dataloader,
    noise_scheduler,
    attack_method=1,
    prediction_type="epsilon",
    sampling_frequency=10,
    ddpm_num_steps=1000,
    membership="mem",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    model.eval()

    save_dir = os.path.join(args.output_name, membership)
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    print(f'gradient_{model_name}.pt')

    def _extract_into_tensor(arr, timesteps, broadcast_shape):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(arr)
        res = arr[timesteps].float().to(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    all_samples_grads = []

    for batch in tqdm.tqdm(dataloader, desc="Extracting Gradients"):
        clean_images = batch[0].to(device)
        clean_images = clean_images.repeat(sampling_frequency, 1, 1, 1)

        noise = torch.randn_like(clean_images)
        # 构建采样 timestep 序列
        timestep_base = torch.tensor(
            [x - 1 for x in range(ddpm_num_steps // sampling_frequency, ddpm_num_steps + 1, ddpm_num_steps // sampling_frequency)],
            device=device
        ).long()  # shape: [sampling_frequency]

        # 重复 timestep，让每个图像都有对应的 timestep
        timesteps = timestep_base.repeat_interleave(clean_images.shape[0] // sampling_frequency)

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noisy_images.requires_grad = True

        model.zero_grad()
        torch.cuda.empty_cache()

        model_output = model(noisy_images, timesteps)

        if attack_method == 1:
            if prediction_type == "epsilon":
                loss = F.mse_loss(model_output, noise)
            elif prediction_type == "sample":
                alpha_t = _extract_into_tensor(
                    noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                )
                snr_weights = alpha_t / (1 - alpha_t)
                loss = snr_weights * F.mse_loss(model_output, clean_images, reduction="none")
                loss = loss.mean()
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

            loss.backward()
            grad_l2 = torch.cat([
                torch.norm(p.grad.detach()).unsqueeze(0)
                for p in model.parameters() if p.grad is not None
            ])
            all_samples_grads.append(grad_l2.unsqueeze(0))
            model.zero_grad()
            torch.cuda.empty_cache()

        elif attack_method == 2:
            grads_per_timestep = []
            for j in range(len(timesteps)):
                if prediction_type == "epsilon":
                    loss = F.mse_loss(model_output[j].unsqueeze(0), noise[j].unsqueeze(0))
                elif prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(model_output, clean_images, reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")

                loss.backward(retain_graph=True)
                grad_l2 = torch.cat([
                    torch.norm(p.grad.detach()).unsqueeze(0)
                    for p in model.parameters() if p.grad is not None
                ])
                grads_per_timestep.append(grad_l2)
                model.zero_grad()
                torch.cuda.empty_cache()

            grads_mean = torch.stack(grads_per_timestep).mean(dim=0)
            all_samples_grads.append(grads_mean.unsqueeze(0))
    all_samples_grads = torch.cat(all_samples_grads)
    torch.save(all_samples_grads, os.path.join(save_dir, f'gradient_{model_name}.pt'))
    return all_samples_grads


def gsamia():
    pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./logs/DDPM_CIFAR10')
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model_type', type=str, default='ddpm')
    parser.add_argument('--ckpt_name', type=str, default='ckpt-step250000.pt')
    parser.add_argument('--k', type=int, default=10)  # 训练组数量
    parser.add_argument('--num_t_groups', type=int, default=1)  # 训练组数量
    parser.add_argument('--sparsity', type=float, default=0.3)  # 训练组数量

    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help=(
            "The directory that save the gradient information."
        ),
    )
    parser.add_argument(
        "--attack_method",
        type=int,
        default=1,
        help=(
            "GSA attack method number."
        ),
    )
    parser.add_argument(
        "--sampling_frequency",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    args = parser.parse_args()

    fix_seed(0)
    ckpt = os.path.join(args.model_dir, args.ckpt_name)
    flag_path = os.path.join(args.model_dir, 'flagfile.txt')
    FLAGS = get_FLAGS(flag_path)
    FLAGS(sys.argv)
    # 判断模型类型
    if args.model_type=='ddpm':
        # Load DDPM
        model = get_model(ckpt, FLAGS, WA=True).to(device)

    model_name = (args.ckpt_name).split('.')[0]
    print(model_name)
    member_loader, nonmember_loader = get_dataset(dataset_root=args.data_root, dataset=args.dataset, batch_size=16, )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=FLAGS.T,
        beta_start=FLAGS.beta_1,
        beta_end=FLAGS.beta_T,
        prediction_type=args.prediction_type
    )

    # 成员
    attack_features_mem = extract_attack_features(
        model=model,
        model_name=model_name,
        dataloader=member_loader,
        noise_scheduler=noise_scheduler,
        attack_method=args.attack_method,
        prediction_type=args.prediction_type,
        sampling_frequency=args.sampling_frequency,
        ddpm_num_steps=FLAGS.T,
        membership="mem",
    )
    # 非成员
    attack_features = extract_attack_features(
        model=model,
        model_name=model_name,
        dataloader=nonmember_loader,
        noise_scheduler=noise_scheduler,
        attack_method=args.attack_method,
        prediction_type=args.prediction_type,
        sampling_frequency=args.sampling_frequency,
        ddpm_num_steps=FLAGS.T,
        membership="nonmem",
    )

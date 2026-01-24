"""
GAN-Leaks: GAN Membership Inference Attack - Core Implementation
Three attack types: FBB (Full Black-Box), PBB (Partial Black-Box), WB (White-Box)
Reference: Hayes et al., "GAN-Leaks: A Taxonomy of Membership Inference Attacks against GANs"
"""

import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Utility Functions
# ============================================================================

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def save_files(save_dir, file_name_list, array_list):
    assert len(file_name_list) == len(array_list)
    for i in range(len(file_name_list)):
        np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)


def get_filepaths_from_dir(data_dir, ext):
    import fnmatch
    pattern = '*.' + ext
    path_list = []
    for d, s, fList in os.walk(data_dir):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                path_list.append(os.path.join(d, filename))
    return sorted(path_list)


def read_image(filepath, resolution=64, cx=89, cy=121):
    import PIL.Image
    img = np.asarray(PIL.Image.open(filepath))
    shape = img.shape

    if shape != (resolution, resolution, 3):
        img = img[cy - 64: cy + 64, cx - 64: cx + 64]
        resize_factor = 128 // resolution
        img = img.astype(np.float32)
        while resize_factor > 1:
            img = (img[0::2, 0::2, :] + img[0::2, 1::2, :] +
                   img[1::2, 0::2, :] + img[1::2, 1::2, :]) * 0.25
            resize_factor -= 1
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    img = img.astype(np.float32) / 255.
    img = img * 2 - 1.
    return img


def inverse_transform(imgs):
    return (imgs + 1.) / 2.


# ============================================================================
# FBB Attack: Full Black-Box Attack
# ============================================================================

class FBBAttack:
    """
    Full Black-Box Attack using k-Nearest Neighbors
    Requires: Access to generated samples only
    """

    def __init__(self, K=5, batch_size=10):
        self.K = K
        self.batch_size = batch_size

    def find_knn(self, nn_obj, query_imgs):
        dist, idx = [], []
        for i in tqdm(range(len(query_imgs) // self.batch_size)):
            x_batch = query_imgs[i * self.batch_size:(i + 1) * self.batch_size]
            x_batch = np.reshape(x_batch, [self.batch_size, -1])
            dist_batch, idx_batch = nn_obj.kneighbors(x_batch, self.K)
            dist.append(dist_batch)
            idx.append(idx_batch)

        try:
            dist, idx = np.concatenate(dist), np.concatenate(idx)
        except:
            dist, idx = np.array(dist), np.array(idx)
        return dist, idx

    def find_pred_z(self, gen_z, idx):
        pred_z = []
        for i in range(len(idx)):
            pred_z.append([gen_z[idx[i, nn]] for nn in range(self.K)])
        return np.array(pred_z)

    def attack(self, gen_imgs, gen_z, pos_query_imgs, neg_query_imgs, save_dir=None):
        # Preprocess generated samples
        gen_feature = np.reshape(gen_imgs, [len(gen_imgs), -1])
        gen_feature = 2. * gen_feature - 1.

        # Build k-NN model
        nn_obj = NearestNeighbors(n_neighbors=self.K, n_jobs=16)
        nn_obj.fit(gen_feature)

        # Positive queries
        pos_loss, pos_idx = self.find_knn(nn_obj, pos_query_imgs)
        pos_z = self.find_pred_z(gen_z, pos_idx)

        # Negative queries
        neg_loss, neg_idx = self.find_knn(nn_obj, neg_query_imgs)
        neg_z = self.find_pred_z(gen_z, neg_idx)

        # Save results
        if save_dir is not None:
            check_folder(save_dir)
            save_files(save_dir, ['pos_loss', 'pos_idx', 'pos_z'], [pos_loss, pos_idx, pos_z])
            save_files(save_dir, ['neg_loss', 'neg_idx', 'neg_z'], [neg_loss, neg_idx, neg_z])

        return pos_loss, neg_loss


# ============================================================================
# PBB Attack: Partial Black-Box Attack
# ============================================================================

class PBBAttack:
    """
    Partial Black-Box Attack by optimizing latent vector z
    Requires: Access to GAN generator interface
    Objective: min λ1*L2 + λ2*LPIPS + λ3*Norm_Reg
    """

    def __init__(self, lambda2=0.2, lambda3=0.001, random_seed=1000):
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.random_seed = random_seed

    def initialize_z(self, init_type, batch_size, z_dim, nn_dir=None):
        if init_type == 'zero':
            z_init = np.zeros((batch_size, z_dim), dtype=np.float32)
        elif init_type == 'random':
            np.random.seed(self.random_seed)
            init_val_np = np.random.normal(size=(z_dim,))
            z_init = np.tile(init_val_np, (batch_size, 1)).astype(np.float32)
        elif init_type == 'nn':
            if nn_dir is None:
                raise ValueError("nn_dir must be provided for 'nn' initialization")
            pos_z = np.load(os.path.join(nn_dir, 'pos_z.npy'))[:, 0, :]
            neg_z = np.load(os.path.join(nn_dir, 'neg_z.npy'))[:, 0, :]
            z_init = {'pos': pos_z, 'neg': neg_z}
        else:
            raise NotImplementedError(f"Initialization type '{init_type}' not implemented")
        return z_init


# ============================================================================
# WB Attack: White-Box Attack
# ============================================================================

class WBAttack:
    """
    White-Box Attack using L-BFGS-B optimizer
    Requires: Access to full GAN model
    Stronger optimization than PBB but higher computational cost
    """

    def __init__(self, lambda2=0.2, lambda3=0.001):
        self.lambda2 = lambda2
        self.lambda3 = lambda3


# ============================================================================
# Evaluation
# ============================================================================

class AttackEvaluator:
    """Evaluate attack performance using ROC/AUC"""

    def plot_roc(self, pos_results, neg_results):
        labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
        results = np.concatenate((neg_results, pos_results))
        fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
        auc = metrics.roc_auc_score(labels, results)
        ap = metrics.average_precision_score(labels, results)
        return fpr, tpr, threshold, auc, ap

    def evaluate_attack(self, pos_loss, neg_loss, attack_name, save_dir=None, reference_load_dir=None):
        plt.figure()

        # Standard attack evaluation
        fpr, tpr, threshold, auc, ap = self.plot_roc(-pos_loss, -neg_loss)
        plt.plot(fpr, tpr, label='%s attack, auc=%.3f, ap=%.3f' % (attack_name, auc, ap))

        # Attack calibration (optional)
        if reference_load_dir is not None:
            pos_ref = np.load(os.path.join(reference_load_dir, 'pos_loss.npy'))
            neg_ref = np.load(os.path.join(reference_load_dir, 'neg_loss.npy'))

            num_pos_samples = np.minimum(len(pos_loss), len(pos_ref))
            num_neg_samples = np.minimum(len(neg_loss), len(neg_ref))

            try:
                pos_calibrate = pos_loss[:num_pos_samples] - pos_ref[:num_pos_samples]
                neg_calibrate = neg_loss[:num_neg_samples] - neg_ref[:num_neg_samples]
            except:
                pos_calibrate = pos_loss[:num_pos_samples] - pos_ref[:num_pos_samples, 0]
                neg_calibrate = neg_loss[:num_neg_samples] - neg_ref[:num_neg_samples, 0]

            fpr, tpr, threshold, auc, ap = self.plot_roc(-pos_calibrate, -neg_calibrate)
            plt.plot(fpr, tpr, label='calibrated %s attack, auc=%.3f, ap=%.3f' % (attack_name, auc, ap))

        plt.legend(loc='lower right')
        plt.xlabel('false positive')
        plt.ylabel('true positive')
        plt.title('ROC curve')

        if save_dir is not None:
            check_folder(save_dir)
            plt.savefig(os.path.join(save_dir, 'roc.png'))
        plt.close()

        return auc, ap


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    print("\nGAN-Leaks Algorithm Summary")
    print("=" * 60)
    print("Attack Types:")
    print("  FBB (Full Black-Box):     Access to generated samples only")
    print("  PBB (Partial Black-Box):  Access to generator interface")
    print("  WB (White-Box):           Access to full GAN model")
    print("\nCore Idea:")
    print("  GANs better reconstruct training samples -> member inference")
    print("  Distance/Loss smaller -> more likely to be a member")
    print("=" * 60)


if __name__ == '__main__':
    print_summary()

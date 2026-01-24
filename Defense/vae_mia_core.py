"""
VAE Membership Inference Attack
支持: 数据加载 | VAE训练 | 三种成员推理攻击
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__} | Keras: {tf.keras.__version__}")

# ===== 配置 =====

class Config:
    """全局配置"""
    def __init__(self):
        self.dataset = "mnist"
        self.batch_size = 128
        self.epochs = 10
        self.learning_rate = 0.001
        self.dim_z = 32
        self.image_size = 28
        self.channels = 1
        self.num_classes = 10

        # 差分隐私
        self.use_dp = False
        self.dp_noise_multiplier = 0.5
        self.dp_l2_norm_clip = 1.0

        # 数据划分
        self.train_size = 10000
        self.test_size = 5000

        # 攻击配置
        self.attack_sample_size = 1000
        self.attack_repetitions = 3

# ===== 数据加载 =====

class DataLoader:
    """数据加载和预处理"""

    def __init__(self, config):
        self.config = config
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

    def load_data(self):
        """加载数据集"""
        print(f"\n{'='*50}\nLoading: {self.config.dataset}\n{'='*50}")

        if self.config.dataset == "mnist":
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            self.config.image_size = 28
            self.config.channels = 1
            self.config.num_classes = 10

        elif self.config.dataset == "fashion_mnist":
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
            self.config.image_size = 28
            self.config.channels = 1
            self.config.num_classes = 10

        elif self.config.dataset == "cifar10":
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            self.config.image_size = 32
            self.config.channels = 3
            self.config.num_classes = 10
            y_train = y_train.flatten()
            y_test = y_test.flatten()

        # 归一化
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # 调整形状
        if self.config.channels == 1:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

        # 采样
        if len(X_train) > self.config.train_size:
            idx = np.random.choice(len(X_train), self.config.train_size, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]

        if len(X_test) > self.config.test_size:
            idx = np.random.choice(len(X_test), self.config.test_size, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

        # 验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = to_categorical(y_train, self.config.num_classes)
        self.y_test = to_categorical(y_test, self.config.num_classes)
        self.X_val = X_val
        self.y_val = to_categorical(y_val, self.config.num_classes)

        print(f"Train: {X_train.shape} | Test: {X_test.shape} | Val: {X_val.shape}\n")

        return self

    def get_data_indices(self):
        """获取数据索引"""
        return {
            "train": np.arange(len(self.X_train)),
            "test": np.arange(len(self.X_test)),
            "val": np.arange(len(self.X_val))
        }

# ===== VAE模型 =====

class Sampling(layers.Layer):
    """重参数化采样层"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(models.Model):
    """条件变分自编码器"""

    def __init__(self, config, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.config = config
        self.dim_z = config.dim_z
        self.num_classes = config.num_classes

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # 损失追踪
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def _build_encoder(self):
        """编码器网络"""
        encoder_inputs = layers.Input(
            shape=(self.config.image_size, self.config.image_size, self.config.channels)
        )
        x = encoder_inputs
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)

        # 潜在空间
        z_mean = layers.Dense(self.dim_z, name="z_mean")(x)
        z_log_var = layers.Dense(self.dim_z, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        # 条件标签
        label_inputs = layers.Input(shape=(self.num_classes,))
        z_with_label = layers.Concatenate()([z, label_inputs])

        return models.Model([encoder_inputs, label_inputs], [z_mean, z_log_var, z_with_label], name="encoder")

    def _build_decoder(self):
        """解码器网络"""
        decoder_inputs = layers.Input(shape=(self.dim_z + self.num_classes,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        outputs = layers.Conv2DTranspose(self.config.channels, 3, activation="sigmoid", padding="same")(x)

        return models.Model(decoder_inputs, outputs, name="decoder")

    def call(self, inputs):
        """前向传播"""
        if isinstance(inputs, list):
            x, y = inputs
        else:
            x, y = inputs, tf.zeros((tf.shape(inputs)[0], self.num_classes))

        z_mean, z_log_var, z_with_label = self.encoder([x, y])
        reconstruction = self.decoder(z_with_label)

        # 损失计算
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(losses.binary_crossentropy(x, reconstruction), axis=[1, 2]))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return reconstruction

    def train_step(self, data):
        """训练步骤"""
        if isinstance(data, list):
            x, y = data
        else:
            x, y = data, tf.zeros((tf.shape(data)[0], self.num_classes))

        with tf.GradientTape() as tape:
            reconstruction = self([x, y], training=True)
            reconstruction_loss = tf.reduce_sum(losses.binary_crossentropy(x, reconstruction), axis=[1, 2])
            z_mean, z_log_var, _ = self.encoder([x, y])
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(tf.reduce_mean(reconstruction_loss))
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """测试步骤"""
        if isinstance(data, list):
            x, y = data
        else:
            x, y = data, tf.zeros((tf.shape(data)[0], self.num_classes))

        reconstruction = self([x, y], training=False)
        reconstruction_loss = tf.reduce_sum(losses.binary_crossentropy(x, reconstruction), axis=[1, 2])
        z_mean, z_log_var, _ = self.encoder([x, y])
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(tf.reduce_mean(reconstruction_loss))
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# ===== 训练器 =====

class VAETrainer:
    """VAE训练器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = {'loss': [], 'val_loss': []}

    def build_model(self):
        """构建模型"""
        print("\n" + "="*50 + "\nBuilding VAE Model\n" + "="*50)
        self.model = VAE(self.config)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.learning_rate))
        print(f"Encoder: {self.model.encoder.input_shape} -> {self.model.encoder.output_shape}")
        print(f"Decoder: {self.model.decoder.input_shape} -> {self.model.decoder.output_shape}\n")

    def train(self, data_loader, save_dir="./models"):
        """训练模型"""
        print("\n" + "="*50 + "\nTraining VAE Model\n" + "="*50)

        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, "vae_weights.h5")
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

        history = self.model.fit(
            [data_loader.X_train, data_loader.y_train],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=([data_loader.X_val, data_loader.y_val],),
            callbacks=[checkpoint],
            verbose=1
        )

        self.history['loss'] = history.history['loss']
        self.history['val_loss'] = history.history['val_loss']

        print("="*50 + f"\nTraining completed!\nTrain Loss: {history.history['loss'][-1]:.4f}\nVal Loss: {history.history['val_loss'][-1]:.4f}\n")

        return self.model

    def load_model(self, weights_path="./models/vae_weights.h5"):
        """加载模型"""
        print(f"\nLoading model from {weights_path}")
        self.build_model()
        self.model.load_weights(weights_path)
        print("Model loaded!\n")

    def generate_samples(self, n_samples=10):
        """生成样本"""
        z_sample = tf.random.normal(shape=(n_samples, self.config.dim_z))
        labels = np.random.randint(0, self.config.num_classes, n_samples)
        y_sample = to_categorical(labels, self.config.num_classes)
        z_with_y = np.concatenate([z_sample, y_sample], axis=1)
        generated = self.model.decoder.predict(z_with_y)
        return generated, labels

# ===== 成员推理攻击 =====

class MembershipInferenceAttacker:
    """成员推理攻击器"""

    def __init__(self, vae_model, encoder_model, decoder_model, config):
        self.vae = vae_model
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.config = config
        self.results = {}

    def reconstruction_attack(self, X_train, X_test, y_train, y_test, repetitions=10):
        """重建攻击 - 通过重建误差判断成员身份"""
        print("\n" + "="*50 + "\nReconstruction Attack\n" + "="*50)

        def compute_error(X, y):
            errors = []
            for i in range(len(X)):
                mse_list = []
                for _ in range(repetitions):
                    z_mean, z_log_var, z_with_label = self.encoder.predict([X[i:i+1], y[i:i+1]], verbose=0)
                    reconstruction = self.decoder.predict(z_with_label, verbose=0)
                    mse_list.append(np.mean((X[i] - reconstruction[0]) ** 2))
                errors.append(np.mean(mse_list))
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(X)}")
            return np.array(errors)

        print("Computing train errors...")
        train_errors = compute_error(X_train, y_train)
        print("\nComputing test errors...")
        test_errors = compute_error(X_test, y_test)

        # 攻击评估
        all_errors = np.concatenate([train_errors, test_errors])
        all_labels = np.concatenate([np.ones(len(train_errors)), np.zeros(len(test_errors))])
        sorted_indices = np.argsort(all_errors)
        predicted_train_indices = set(sorted_indices[:len(train_errors)])
        actual_train_indices = set(range(len(train_errors)))
        accuracy = len(predicted_train_indices & actual_train_indices) / len(train_errors)

        fpr, tpr, _ = metrics.roc_curve(all_labels, -all_errors)
        auc_score = metrics.auc(fpr, tpr)

        self.results['reconstruction_attack'] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'mean_train_error': np.mean(train_errors),
            'mean_test_error': np.mean(test_errors),
            'successful_set_attack': np.mean(train_errors) < np.mean(test_errors)
        }

        print(f"\nAccuracy: {accuracy:.4f} | AUC: {auc_score:.4f}")
        print(f"Train Error: {np.mean(train_errors):.6f} | Test Error: {np.mean(test_errors):.6f}\n")

        return self.results['reconstruction_attack']

    def monte_carlo_pca_attack(self, X_train, X_test, y_train, y_test, n_components=50,
                               n_generated=500, percentile=10):
        """蒙特卡洛PCA攻击 - 通过生成样本距离推断成员"""
        print("\n" + "="*50 + "\nMonte Carlo PCA Attack\n" + "="*50)

        # PCA变换
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        print("Training PCA...")
        pca = PCA(n_components=n_components)
        pca.fit(X_test_flat)
        X_train_pca = pca.transform(X_train_flat)
        X_test_pca = pca.transform(X_test_flat)

        # 生成样本
        print(f"Generating {n_generated} samples...")
        z_samples = np.random.normal(0, 1, (n_generated, self.config.dim_z))
        y_samples = to_categorical(np.random.randint(0, self.config.num_classes, n_generated), self.config.num_classes)
        z_with_y = np.concatenate([z_samples, y_samples], axis=1)
        generated = self.decoder.predict(z_with_y, verbose=0)
        generated_pca = pca.transform(generated.reshape(n_generated, -1))

        # 距离计算
        print("Computing distances...")
        train_distances = cdist(X_train_pca, generated_pca, 'euclidean')
        test_distances = cdist(X_test_pca, generated_pca, 'euclidean')

        epsilon = np.percentile(np.concatenate([train_distances.flatten(), test_distances.flatten()]), percentile)

        # 积分近似
        def compute_integral(distances, eps):
            return np.array([np.sum(d < eps) / len(d) for d in distances])

        train_scores = compute_integral(train_distances, epsilon)
        test_scores = compute_integral(test_distances, epsilon)

        # 攻击评估
        all_scores = np.concatenate([train_scores, test_scores])
        all_labels = np.concatenate([np.ones(len(train_scores)), np.zeros(len(test_scores))])
        sorted_indices = np.argsort(all_scores)[::-1]
        predicted_train_indices = set(sorted_indices[:len(train_scores)])
        actual_train_indices = set(range(len(train_scores)))
        accuracy = len(predicted_train_indices & actual_train_indices) / len(train_scores)

        fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
        auc_score = metrics.auc(fpr, tpr)

        self.results['mc_pca_attack'] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'mean_train_score': np.mean(train_scores),
            'mean_test_score': np.mean(test_scores),
            'epsilon': epsilon,
            'successful_set_attack': np.sum(train_scores) > np.sum(test_scores)
        }

        print(f"\nAccuracy: {accuracy:.4f} | AUC: {auc_score:.4f}")
        print(f"Train Score: {np.mean(train_scores):.6f} | Test Score: {np.mean(test_scores):.6f}\n")

        return self.results['mc_pca_attack']

    def ssim_attack(self, X_train, X_test, y_train, y_test, repetitions=5):
        """SSIM攻击 - 通过结构相似性推断成员"""
        print("\n" + "="*50 + "\nSSIM Attack\n" + "="*50)

        multichannel = self.config.channels > 1

        def compute_ssim(X, y):
            ssim_values = []
            for i in range(len(X)):
                _, _, z_with_label = self.encoder.predict([X[i:i+1], y[i:i+1]], verbose=0)
                reconstruction = self.decoder.predict(z_with_label, verbose=0)[0]

                if multichannel:
                    ssim_val = ssim(X[i], reconstruction, multichannel=True, channel_axis=2)
                else:
                    ssim_val = ssim(X[i, ..., 0], reconstruction[..., 0])

                ssim_values.append(ssim_val)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(X)}")
            return np.array(ssim_values)

        print("Computing train SSIM...")
        train_ssim = compute_ssim(X_train, y_train)
        print("\nComputing test SSIM...")
        test_ssim = compute_ssim(X_test, y_test)

        # 攻击评估
        all_ssim = np.concatenate([train_ssim, test_ssim])
        all_labels = np.concatenate([np.ones(len(train_ssim)), np.zeros(len(test_ssim))])
        sorted_indices = np.argsort(all_ssim)[::-1]
        predicted_train_indices = set(sorted_indices[:len(train_ssim)])
        actual_train_indices = set(range(len(train_ssim)))
        accuracy = len(predicted_train_indices & actual_train_indices) / len(train_ssim)

        fpr, tpr, _ = metrics.roc_curve(all_labels, all_ssim)
        auc_score = metrics.auc(fpr, tpr)

        self.results['ssim_attack'] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'mean_train_ssim': np.mean(train_ssim),
            'mean_test_ssim': np.mean(test_ssim),
            'successful_set_attack': np.mean(train_ssim) > np.mean(test_ssim)
        }

        print(f"\nAccuracy: {accuracy:.4f} | AUC: {auc_score:.4f}")
        print(f"Train SSIM: {np.mean(train_ssim):.6f} | Test SSIM: {np.mean(test_ssim):.6f}\n")

        return self.results['ssim_attack']

    def save_results(self, filepath="./results/attack_results.json"):
        """保存结果"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_serializable = {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in self.results.items()}

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to {filepath}")

    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*50 + "\nATTACK RESULTS SUMMARY\n" + "="*50)
        for attack_name, results in self.results.items():
            print(f"\n{attack_name.upper()}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  AUC: {results['auc']:.4f}")
            print(f"  Successful Set Attack: {results['successful_set_attack']}")
        print("\n" + "="*50 + "\n")

# ===== 主流程 =====

def main():
    """主流程"""
    print("\n" + "="*70)
    print(" " * 15 + "VAE MEMBERSHIP INFERENCE ATTACK")
    print(" " * 20 + "Core Implementation")
    print("="*70)

    # 配置
    config = Config()
    print("\n[Configuration]")
    print(f"  Dataset: {config.dataset} | Train: {config.train_size} | Test: {config.test_size}")
    print(f"  Epochs: {config.epochs} | Batch: {config.batch_size} | Latent: {config.dim_z}")

    # 数据
    data_loader = DataLoader(config).load_data()

    # 训练
    trainer = VAETrainer(config)
    trainer.build_model()
    model = trainer.train(data_loader, save_dir="./models")

    # 攻击
    attacker = MembershipInferenceAttacker(model, model.encoder, model.decoder, config)

    sample_size = config.attack_sample_size
    print(f"\n[Attack Data] Sample size: {sample_size}")

    X_attack_train = data_loader.X_train[:sample_size]
    y_attack_train = data_loader.y_train[:sample_size]
    X_attack_test = data_loader.X_test[:sample_size]
    y_attack_test = data_loader.y_test[:sample_size]

    print("\n" + "="*70 + "\n" + " " * 25 + "EXECUTING ATTACKS" + "\n" + "="*70)

    attacker.reconstruction_attack(X_attack_train, X_attack_test, y_attack_train, y_attack_test, repetitions=5)
    attacker.monte_carlo_pca_attack(X_attack_train, X_attack_test, y_attack_train, y_attack_test, n_components=50, n_generated=500, percentile=10)
    attacker.ssim_attack(X_attack_train, X_attack_test, y_attack_train, y_attack_test, repetitions=3)

    # 结果
    attacker.save_results("./results/attack_results.json")
    attacker.print_summary()

    print("="*70 + "\n" + " " * 20 + "EXPERIMENT COMPLETED" + "\n" + "="*70 + "\n")

    return attacker.results

# ===== 使用示例 =====

if __name__ == "__main__":
    results = main()

    print("\n" + "="*70 + "\nUSAGE EXAMPLES:\n" + "="*70)
    print("""
# 1. 仅训练模型
config = Config()
data_loader = DataLoader(config).load_data()
trainer = VAETrainer(config)
trainer.build_model()
model = trainer.train(data_loader)

# 2. 加载模型并攻击
config = Config()
trainer = VAETrainer(config)
trainer.load_model("./models/vae_weights.h5")
attacker = MembershipInferenceAttacker(trainer.model, trainer.model.encoder, trainer.model.decoder, config)
attacker.reconstruction_attack(X_train, X_test, y_train, y_test)
attacker.monte_carlo_pca_attack(X_train, X_test, y_train, y_test)
attacker.ssim_attack(X_train, X_test, y_train, y_test)

# 3. 生成样本
samples, labels = trainer.generate_samples(10)
    """)
    print("="*70 + "\n")

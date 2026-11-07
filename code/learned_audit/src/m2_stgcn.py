"""M2: Deb et al. 2022 (TNSRE) ST-GCN+LSTM regressor (fokhruli/STGCN-rehab).

TensorFlow/Keras. Vendored from third_party/stgcn_rehab/GCN/sgcn_lstm.py with
two mechanical changes for a 19-joint skeleton:
  * constructor arg num_joints replaces the two hardcoded filters=25 in the
    two ConvLSTM2D attention layers (joints count), nothing else touched.
Graph AD/AD2/bias built from ec3d BONES via their normalize_adjacency logic.
Recipe per repo defaults (train.py): Huber(delta=0.1), Adam(lr=1e-4),
epochs=1000, batch_size=10, validation_split=0.2 (on a seed-shuffled array,
as in the repo where fit() receives train_test_split output), ModelCheckpoint
callback (weights only under Keras 2.15; never reloaded). StandardScaler on
X and y fit on the training persons. Score: regression output
inverse-transformed to label units (error=1, correct=0). Decision: output >
0.5. Last-epoch weights (their train.py predicts with the in-memory model
after fit; the checkpoint file is never reloaded for prediction).
Env: tensorflow 2.15.1 / keras 2.15.0 (repo pins TF 2.4.0; no py3.10 build).
"""
import numpy as np

NUM_TIMESTEP = 100
EPOCHS = 1000
BATCH_SIZE = 10
LR = 1e-4


def build_graph(bones, num_joints=19):
    """Graph.normalize_adjacency from their graph.py with our edge list."""
    import tensorflow as tf
    self_link = [(i, i) for i in range(num_joints)]
    edge = self_link + list(bones)
    A = np.zeros((num_joints, num_joints))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    A2 = np.zeros((num_joints, num_joints))
    for root in range(A.shape[1]):
        for neighbour in range(A.shape[0]):
            if A[root, neighbour] == 1:
                for nn2 in range(A.shape[0]):
                    if A[neighbour, nn2] == 1:
                        A2[root, nn2] = 1
    bias_mat_1 = np.where(A != 0, np.zeros(A.shape), -1e9).astype("float32")
    bias_mat_2 = np.where(A2 != 0, A2, -1e9).astype("float32")
    return (tf.convert_to_tensor(A.astype("float32")),
            tf.convert_to_tensor(A2.astype("float32")),
            tf.convert_to_tensor(bias_mat_1),
            tf.convert_to_tensor(bias_mat_2))


class _Scaler:
    """sklearn StandardScaler equivalent (zero-variance -> scale 1)."""

    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu_) / self.sd_

    def inverse_transform(self, X):
        return X * self.sd_ + self.mu_


class SgcnLstm:
    """Vendored GCN/sgcn_lstm.py with num_joints parameterized."""

    def __init__(self, AD, AD2, bias_mat_1, bias_mat_2, num_joints=19,
                 lr=LR, epoach=EPOCHS, batch_size=BATCH_SIZE):
        self.AD, self.AD2 = AD, AD2
        self.bias_mat_1, self.bias_mat_2 = bias_mat_1, bias_mat_2
        self.lr, self.epoach, self.batch_size = lr, epoach, batch_size
        self.num_joints = num_joints

    def sgcn(self, Input):
        import tensorflow as tf
        from tensorflow.keras.layers import Conv2D, ConvLSTM2D, Dropout, concatenate
        k1 = Conv2D(64, (9, 1), padding="same", activation="relu")(Input)
        k = concatenate([Input, k1], axis=-1)

        x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation="relu")(k)
        expand_x1 = tf.expand_dims(x1, axis=3)
        f_1 = ConvLSTM2D(filters=self.num_joints, kernel_size=(1, 1),
                         return_sequences=True)(expand_x1)
        f_1 = f_1[:, :, :, 0, :]
        logits = f_1
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)
        gcn_x1 = tf.keras.layers.Lambda(
            lambda x: tf.einsum("ntvw,ntwc->ntvc", x[0], x[1]))([coefs, x1])

        y1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation="relu")(k)
        expand_y1 = tf.expand_dims(y1, axis=3)
        f_2 = ConvLSTM2D(filters=self.num_joints, kernel_size=(1, 1),
                         return_sequences=True)(expand_y1)
        f_2 = f_2[:, :, :, 0, :]
        logits = f_2
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(
            lambda x: tf.einsum("ntvw,ntwc->ntvc", x[0], x[1]))([coefs, y1])

        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)

        z1 = Conv2D(16, (9, 1), padding="same", activation="relu")(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = Conv2D(16, (15, 1), padding="same", activation="relu")(z1)
        z2 = Dropout(0.25)(z2)
        z3 = Conv2D(16, (20, 1), padding="same", activation="relu")(z2)
        z3 = Dropout(0.25)(z3)
        return concatenate([z1, z2, z3], axis=-1)

    def Lstm(self, x):
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[2] * x.shape[3]))(x)
        rec = LSTM(80, return_sequences=True)(x)
        rec = Dropout(0.25)(rec)
        rec1 = LSTM(40, return_sequences=True)(rec)
        rec1 = Dropout(0.25)(rec1)
        rec2 = LSTM(40, return_sequences=True)(rec1)
        rec2 = Dropout(0.25)(rec2)
        rec3 = LSTM(80)(rec2)
        rec3 = Dropout(0.25)(rec3)
        return Dense(1, activation="linear")(rec3)

    def build(self, train_x):
        import tensorflow as tf
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        seq_input = Input(shape=(None, train_x.shape[2], train_x.shape[3]), batch_size=None)
        x = self.sgcn(seq_input)
        y = self.sgcn(x)
        y = y + x
        z = self.sgcn(y)
        z = z + y
        out = self.Lstm(z)
        self.model = Model(seq_input, out)
        # repo: experimental_steps_per_execution=50 (TF 2.4 name); Keras 2.15
        # maps it to steps_per_execution with a deprecation warning.
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1),
                           steps_per_execution=50,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

    def fit(self, train_x, train_y, ckpt_path, log=print):
        import tensorflow as tf
        from tensorflow.keras.callbacks import ModelCheckpoint
        # repo: ModelCheckpoint("best model ex4/best_model.hdf5", monitor=val_loss,
        # save_best_only). Keras 2.15 cannot serialize the einsum Lambda layers
        # to a full HDF5 model (closure over the tf module), so weights only.
        # The checkpoint is a side effect either way: the repo predicts with the
        # in-memory model after fit and never reloads it (last-epoch weights).
        checkpoint = ModelCheckpoint(ckpt_path, monitor="val_loss",
                                     save_best_only=True, mode="auto",
                                     save_weights_only=True)
        cb = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, l: log("m2 epoch {}/{} loss {:.4f} val_loss {:.4f}".format(
                e + 1, self.epoach, l.get("loss", float("nan")), l.get("val_loss", float("nan"))))
            if (e + 1) % 100 == 0 or e == 0 else None)
        return self.model.fit(train_x, train_y, validation_split=0.2,
                              epochs=self.epoach, batch_size=self.batch_size,
                              callbacks=[checkpoint, cb], verbose=0)

    def predict(self, data):
        return self.model.predict(data, verbose=0)


def train_model(train_items, bones, seed, run_dir, log=print):
    import os
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)

    X = np.stack([x for _, x in train_items]).astype("float32")  # (n,100,19,3)
    n, T, V, C = X.shape
    # Scalers fit on the training persons only (the repo fits on its whole
    # file before its own split; here the held-out person must stay unseen).
    sc1 = _Scaler().fit(X.reshape(-1, V * C))
    y = np.array([s["err"] for s, _ in train_items], dtype="float32").reshape(-1, 1)
    sc2 = _Scaler().fit(y)
    train_x = sc1.transform(X.reshape(-1, V * C)).reshape(n, T, V, C).astype("float32")
    train_y = sc2.transform(y).astype("float32")
    # Keras validation_split takes the LAST 20% of the array as given. In the
    # repo the array reaching fit() has already been shuffled by
    # train_test_split(random_state=42), so that 20% is a random subset; our
    # train_items arrive sorted by (action, person, label, rep), so shuffle
    # once with the run seed before fit to keep the repo's effective recipe.
    perm = np.random.default_rng(seed).permutation(n)
    train_x, train_y = train_x[perm], train_y[perm]

    AD, AD2, b1, b2 = build_graph(bones, V)
    algo = SgcnLstm(AD, AD2, b1, b2, num_joints=V)
    algo.build(train_x)
    ckpt_dir = os.path.join(run_dir, "best model ex4")
    os.makedirs(ckpt_dir, exist_ok=True)
    algo.fit(train_x, train_y, os.path.join(ckpt_dir, "best_model.hdf5"), log=log)
    return algo, sc1, sc2


def score_items(bundle, items):
    algo, sc1, sc2 = bundle
    if not items:
        return []
    X = np.stack([x for _, x in items]).astype("float32")
    n, T, V, C = X.shape
    Xt = sc1.transform(X.reshape(-1, V * C)).reshape(n, T, V, C).astype("float32")
    y_pred = sc2.inverse_transform(algo.predict(Xt))
    rows = []
    for (s, _), yp in zip(items, y_pred):
        score = float(np.asarray(yp).reshape(-1)[0])
        rows.append((s, score, int(score > 0.5)))
    return rows

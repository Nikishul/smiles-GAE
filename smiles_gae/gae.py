import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GAE:
    def __init__(self, input_shape, output_shape, latent_dim, lstm_dim):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim

        self.build_autoencoder()

    def build_autoencoder(self):
        unroll = False
        self.encoder_inputs = Input(shape=self.input_shape, name="Smiles_input")
        self.energy_inputs = Input(shape=1, name="Energy_input")
        encoder = LSTM(self.lstm_dim, return_state=True, unroll=unroll, name="Encoder_LSTM")
        self.encoder_outputs, self.state_h, self.state_c = encoder(self.encoder_inputs)
        self.states = Concatenate(axis=-1, name="States_concatenation")([self.state_h, self.state_c])
        self.neck = Dense(self.latent_dim, activation="relu", name="Encoder_neck")
        self.neck_outputs = self.neck(self.states)
        self.energy_embed = Concatenate(axis=-1, name="Energy_concatenation")([self.neck_outputs, self.energy_inputs])

        self.decode_h = Dense(self.lstm_dim, activation="relu", name="State_h_decoder")
        self.decode_c = Dense(self.lstm_dim, activation="relu", name="State_c_decoder")
        self.state_h_decoded = self.decode_h(self.energy_embed)
        self.state_c_decoded = self.decode_c(self.energy_embed)
        self.encoder_states = [self.state_h_decoded, self.state_c_decoded]
        self.decoder_inputs = Input(shape=self.input_shape, name="Decoder_input")
        self.decoder_lstm = LSTM(self.lstm_dim, return_sequences=True, unroll=unroll, name="Decoder_LSTM")
        self.decoder_outputs = self.decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)
        self.decoder_dense = Dense(self.output_shape, activation="softmax", name="Decoder_output")
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        self.ae_model = Model([self.encoder_inputs, self.decoder_inputs, self.energy_inputs], self.decoder_outputs)
        self.ae_model.summary()

    def build_complementary_models(self):
        self.smiles_to_latent_model = Model(self.encoder_inputs, self.neck_outputs)

        self.latent_input = Input(shape=(self.latent_dim,))
        self.latent_energy_input = Input(shape=(1))
        # reuse_layers

        sample_latent_input = Concatenate(axis=-1)([self.latent_input, self.latent_energy_input])
        self.state_h_decoded_2 = self.decode_h(sample_latent_input)
        self.state_c_decoded_2 = self.decode_c(sample_latent_input)
        self.latent_to_states_model = Model(
            [self.latent_input, self.latent_energy_input], [self.state_h_decoded_2, self.state_c_decoded_2]
        )

        self.inf_decoder_inputs = Input(batch_shape=(30000, 1, self.input_shape[1]))
        self.inf_decoder_lstm = LSTM(self.lstm_dim, return_sequences=True, unroll=False, stateful=True)
        self.inf_decoder_outputs = self.inf_decoder_lstm(self.inf_decoder_inputs)
        self.inf_decoder_dense = Dense(self.output_shape, activation="softmax")
        self.inf_decoder_outputs = self.inf_decoder_dense(self.inf_decoder_outputs)
        self.sample_model = Model(self.inf_decoder_inputs, self.inf_decoder_outputs)

        for i in range(1, 3):
            self.sample_model.layers[i].set_weights(self.ae_model.layers[i + 8].get_weights())

    def train(
        self,
        X_train,
        energy_train,
        Y_train,
        X_test,
        Y_test,
        epochs=160,
        batch_size=256,
        start_learning_rate=0.005,
        logdir="logs",
    ):

        opt = Adam(learning_rate=start_learning_rate)
        self.ae_model.compile(optimizer=opt, loss="categorical_crossentropy")

        self.h = History()
        self.rlr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1, epsilon=1e-4
        )
        self.es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        self.ae_model.fit(
            [X_train, X_train, energy_train],
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[self.h, self.rlr, self.es, self.tensorboard_callback],
            validation_split=0.1,
            validation_data=([X_test, X_test], Y_test),
        )
        self.build_complementary_models()
        self.save_all_models()

        return self

    def latent_to_smiles_batch(self, latent_batch, energy_batch):
        batch_size = latent_batch.shape[0]
        states = self.latent_to_states_model.predict([latent_batch, energy_batch])
        self.sample_model.layers[1].reset_states(states=[states[0], states[1]])

        startidx = self.char_to_int["!"]
        samplevec = np.zeros((batch_size, 1, self.n_unique_characters))
        samplevec[:, 0, startidx] = 1
        smiles_list = [""] * batch_size
        batch_set = set(range(0, batch_size))

        for i in range(self.embed):
            batch_out = self.sample_model.predict(samplevec, batch_size=batch_size)
            samplevec = np.zeros((batch_size, 1, self.n_unique_characters))
            set_min = set()
            for j in batch_set:
                out_vector = batch_out[j]
                sampleidx = np.argmax(out_vector)
                samplechar = self.int_to_char[sampleidx]
                if samplechar == "E":
                    set_min.add(j)
                    continue
                smiles_list[j] = smiles_list[j] + samplechar
                samplevec[j, 0, sampleidx] = 1
            batch_set -= set_min
            if not batch_set:
                break

        return smiles_list

    def save_all_models(self, default_path="saved_models"):
        if not os.path.exists(default_path):
            os.mkdir(default_path)

        default_folder = Path(default_path)
        self.ae_model.save(default_folder / "lstm_autoenc_energy.h5")
        self.smiles_to_latent_model.save(default_folder / "energy_smi2lat.h5")
        self.latent_to_states_model.save(default_folder / "energy_lat2state.h5")
        self.sample_model.save(default_folder / "energy_potent_sample_model.h5")

    def load_all_models(self, path="saved_models"):
        folder = Path(path)

        self.ae_model = tf.keras.models.load_model(folder / "lstm_autoenc_energy.h5")
        self.smiles_to_latent_model = tf.keras.models.load_model(folder / "energy_smi2lat.h5")
        self.latent_to_states_model = tf.keras.models.load_model(folder / "energy_lat2state.h5")
        self.sample_model = tf.keras.models.load_model(folder / "energy_potent_sample_model.h5")

    def generate(self, embed, orig_smiles, orig_energy, scales):

        orig_smiles = orig_smiles.reset_index(drop=True)
        orig_energy = orig_energy.reset_index(drop=True)
        total_new_smis = []
        coef = 0.5
        multiplier = 1.05
        return_df = pd.DataFrame(columns=["source_smiles", "source_energy"])
        counter = 0

        while counter < 10000:
            coef *= multiplier
            random_inx = np.random.choice(len(embed), size=30000)
            new_embed = embed[random_inx]
            new_embed = new_embed + [np.array([np.random.normal(0, coef * y) for y in scales]) for x in range(30000)]

            new_smis = self.latent_to_smiles_batch(new_embed, orig_energy[random_inx])
            for new_smi in new_smis:
                try:
                    mol = Chem.MolFromSmiles(new_smi)
                    if mol:
                        if len(new_smi) < 10:
                            total_new_smis.append(None)
                        else:
                            total_new_smis.append(new_smi)
                    else:
                        total_new_smis.append(None)
                except:
                    total_new_smis.append(None)

            buf = pd.DataFrame(
                {
                    "source_smiles": orig_smiles[random_inx],
                    "source_energy": orig_energy[random_inx],
                    "generated_energy": orig_energy[random_inx] - 1,
                }
            )
            return_df = return_df.append(buf.copy())

            counter = len(list(set([x for x in total_new_smis if x is not None])))
        return_df["gener_smiles"] = total_new_smis
        return_df.drop_duplicates(subset=["gener_smiles"], inplace=True)

        return return_df

    def generate_energy(self, smiles, energies, improvement=0.5):
        total_new_smis = []
        for smi, energy in tqdm(zip(smiles, energies), position=0, total=len(energies)):
            smi = smi.reshape((1, 153, 35))
            energy = np.array(energy).reshape((-1, 1))
            embed = self.smiles_to_latent_model.predict(smi)

            new_smi = self.latent_to_smiles(embed, energy - improvement)
            try:
                mol = Chem.MolFromSmiles(new_smi)
                if mol:
                    total_new_smis.append(new_smi)
                else:
                    total_new_smis.append([])
            except:
                total_new_smis.append([])
        return total_new_smis

    def generate_random(self, energy, locs, scales):
        total_new_smis = []
        coef = 0.5
        multiplier = 1.05
        while len(total_new_smis) < 10000:

            new_embed = np.array(
                [np.array([np.random.normal(x, coef * y) for x, y in zip(locs, scales)]) for z in range(30000)]
            )
            coef *= multiplier
            random_inx = np.random.choice(len(new_embed), size=30000)
            new_embed = new_embed[random_inx]

            # new_embed = new_embed.reshape((7770, 1, 64))
            energy_batch = np.ones(len(new_embed))
            energy_batch.fill(energy)
            new_smis = self.latent_to_smiles_batch(new_embed, energy_batch)

            for new_smi in new_smis:
                try:
                    mol = Chem.MolFromSmiles(new_smi)
                    if mol:
                        if len(new_smi) >= 10:
                            total_new_smis.append(new_smi)
                except:
                    pass

        total_new_smis = list(set(total_new_smis))
        return total_new_smis

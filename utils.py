import numpy as np, tensorflow as tf
from scipy.signal import resample

class callCompetition():
    def __init__(self, sbjIdx):
        self.sbjIdx = sbjIdx
        self.path = "/home/ko/Desktop/pycharm-2018.3.5/projects/Data/BCIC_IV_2A/preprocessed_eeg"

    def prepareData(self):
        Xtr = np.load(self.path + "/A{:02d}T.npy".format(self.sbjIdx))
        Ytr = np.load(self.path + "/A{:02d}T_label.npy".format(self.sbjIdx)) - 1
        Xts = np.load(self.path + "/A{:02d}E.npy".format(self.sbjIdx))
        Yts = np.load(self.path + "/A{:02d}E_label.npy".format(self.sbjIdx)) - 1

        Ytr = np.eye(np.unique(Ytr).shape[0])[Ytr]
        Yts = np.eye(np.unique(Yts).shape[0])[Yts]

        # Xtr, Xts = resample(Xtr, int(Xtr.shape[1] * 0.4), axis=1), resample(Xts, int(Xts.shape[1] * 0.4), axis=1)

        Xtr, Xts = np.expand_dims(np.moveaxis(Xtr, -1, 0), -1), np.expand_dims(np.moveaxis(Xts, -1, 0), -1)
        return Xtr, Ytr, Xtr, Ytr, Xts, Yts

class callDataset_sbj_indep():
    def __init__(self, trgsbjIdx):
        self.trgsbjIdx = trgsbjIdx
        assert 0 < trgsbjIdx < 55, "We only have subject 1 to 54."
        srcsbjIdcs = np.setdiff1d(np.linspace(1, 54, 54, dtype=np.int), trgsbjIdx)

        self.path = "/home/ko/Desktop/pycharm-2018.3.5/projects/Data/KU_SW_Lee/preprocessed"

        self.chSelection = [7, 32, 8, 9, 33, 10,
                            34, 12, 35, 13, 36, 14, 37,
                            17, 38, 18, 39, 19, 40, 20]

        self.vldsbjIdx = np.random.permutation(srcsbjIdcs.shape[0])[0]
        self.srcsbjIdcs = np.setdiff1d(srcsbjIdcs, self.vldsbjIdx)

    def loadData(self):
        Xtrg = np.load(self.path + "/TIME_Sess02_sub{:>02d}_test.npy".format(self.trgsbjIdx))[self.chSelection, 500:, :]
        Ytrg = np.load(self.path + "/TIME_Sess02_sub{:>02d}_tslbl.npy".format(self.trgsbjIdx))

        Xvld = np.load(self.path + "/TIME_Sess02_sub{:>02d}_test.npy".format(self.vldsbjIdx))[self.chSelection, 500:, :]
        Yvld = np.load(self.path + "/TIME_Sess02_sub{:>02d}_tslbl.npy".format(self.vldsbjIdx))



        a = np.array([np.load(self.path + "/TIME_Sess01_sub{:>02d}_train.npy".format(i))[self.chSelection, 500:, :] for i in self.srcsbjIdcs])
        b = np.array([np.load(self.path + "/TIME_Sess01_sub{:>02d}_test.npy".format(i))[self.chSelection, 500:, :] for i in self.srcsbjIdcs])
        c = np.array([np.load(self.path + "/TIME_Sess02_sub{:>02d}_train.npy".format(i))[self.chSelection, 500:, :] for i in self.srcsbjIdcs])
        d = np.array([np.load(self.path + "/TIME_Sess02_sub{:>02d}_test.npy".format(i))[self.chSelection, 500:, :] for i in self.srcsbjIdcs])
        Xsrc = np.concatenate((a, b, c, d), axis=-1)

        a = np.array([np.load(self.path + "/TIME_Sess01_sub{:>02d}_trlbl.npy".format(i)) for i in self.srcsbjIdcs])
        b = np.array([np.load(self.path + "/TIME_Sess01_sub{:>02d}_tslbl.npy".format(i)) for i in self.srcsbjIdcs])
        c = np.array([np.load(self.path + "/TIME_Sess02_sub{:>02d}_trlbl.npy".format(i)) for i in self.srcsbjIdcs])
        d = np.array([np.load(self.path + "/TIME_Sess02_sub{:>02d}_tslbl.npy".format(i)) for i in self.srcsbjIdcs])
        Ysrc = np.concatenate((a, b, c, d), axis=-1)

        Xsrc = np.moveaxis(Xsrc, 0, -1)
        Ysrc = np.moveaxis(Ysrc, 0, -1)

        # Downsample to 100Hz sampling rate
        Xsrc = resample(Xsrc, int(Xsrc.shape[1] * 0.1), axis=1)
        Xvld = resample(Xvld, int(Xvld.shape[1] * 0.1), axis=1)
        Xtrg = resample(Xtrg, int(Xtrg.shape[1] * 0.1), axis=1)

        Xsrc = np.reshape(Xsrc, newshape=(Xsrc.shape[0], Xsrc.shape[1], -1))
        Ysrc = np.reshape(Ysrc, newshape=(Ysrc.shape[0], -1))

        return Xsrc, Ysrc, Xvld, Yvld, Xtrg, Ytrg

    def prepareData(self):
        Xsrc, Ysrc, Xvld, Yvld, Xtrg, Ytrg = self.loadData()
        Xsrc, Xvld, Xtrg = np.moveaxis(Xsrc, -1, 0), np.moveaxis(Xvld, -1, 0), np.moveaxis(Xtrg, -1, 0)
        Xsrc, Xvld, Xtrg = np.expand_dims(Xsrc, -1), np.expand_dims(Xvld, -1), np.expand_dims(Xtrg, -1)
        Ysrc, Yvld, Ytrg = np.moveaxis(Ysrc, -1, 0), np.moveaxis(Yvld, -1, 0), np.moveaxis(Ytrg, -1, 0)
        return Xsrc, Ysrc, Xvld, Yvld, Xtrg, Ytrg




class callDataset():
    def __init__(self, sbjIdx, sessIdx):
        assert 0 < sbjIdx < 55, "We only have subject 1 to 54."
        assert 0 < sessIdx < 3, "We only have session 1 and 2."

        self.sbjIdx, self.sessIdx = sbjIdx, sessIdx
        self.path = "/home/ko/Desktop/pycharm-2018.3.5/projects/Data/KU_SW_Lee/preprocessed"

        self.chSelection = [7, 32, 8, 9, 33, 10,
                            34, 12, 35, 13, 36, 14, 37,
                            17, 38, 18, 39, 19, 40, 20]

    def loadData(self):
        Xtr = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_train.npy"
                      .format(self.sessIdx, self.sbjIdx))[self.chSelection, 500:, :]
        Ytr = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_trlbl.npy".format(self.sessIdx, self.sbjIdx))

        Xts = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_test.npy"
                      .format(self.sessIdx, self.sbjIdx))[self.chSelection, 500:, :]
        Yts = np.load(self.path + "/TIME_Sess{:>02d}_sub{:>02d}_tslbl.npy".format(self.sessIdx, self.sbjIdx))

        # Downsample to 100Hz sampling rate
        Xtr, Xts = resample(Xtr, int(Xtr.shape[1] * 0.1), axis=1), resample(Xts, int(Xts.shape[1] * 0.1), axis=1)

        # Divide validation set
        numVals = int(Xtr.shape[-1]/10)
        Xvl, Yvl = Xtr[:, :, :numVals], Ytr[:, :numVals]
        Xtr, Ytr = Xtr[:, :, numVals:], Ytr[:, numVals:]

        return Xtr, Ytr, Xvl, Yvl, Xts, Yts

    def GaussNorm(self):
        Xtr, Ytr, Xvl, Yvl, Xts, Yts = self.loadData()
        # Calculate mean and standard deviation for Gaussian normalization
        meantr, stdtr = np.mean(Xtr, axis=(1, 2), keepdims=True), np.std(Xtr, axis=(1, 2), keepdims=True)

        def myNorm(X): return (X - meantr)/stdtr

        Xtr, Xvl, Xts = myNorm(Xtr), myNorm(Xvl), myNorm(Xts)

        return Xtr, Ytr, Xvl, Yvl, Xts, Yts

    def prepareData(self, is_GaussNorm=True):
        if is_GaussNorm: Xtr, Ytr, Xvl, Yvl, Xts, Yts = self.GaussNorm()
        else: Xtr, Ytr, Xvl, Yvl, Xts, Yts = self.loadData()

        Xtr, Xvl, Xts = np.moveaxis(Xtr, -1, 0), np.moveaxis(Xvl, -1, 0), np.moveaxis(Xts, -1, 0)
        Xtr, Xvl, Xts = np.expand_dims(Xtr, -1), np.expand_dims(Xvl, -1), np.expand_dims(Xts, -1)
        Ytr, Yvl, Yts = np.moveaxis(Ytr, 0, 1), np.moveaxis(Yvl, 0, 1), np.moveaxis(Yts, 0, 1)

        return Xtr, Ytr, Xvl, Yvl, Xts, Yts




def gradient(model, inputs, labels):
    with tf.GradientTape() as tape:
        yhat = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(labels, yhat, label_smoothing=.1)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad
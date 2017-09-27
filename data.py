import h5py
import numpy as np
import random
from scipy.io import loadmat


class Unwrapdata(object):

    def __init__(self, subject, roi):
        self.dataset = ""
        self.subject = subject
        self.roi = roi
        self.indexes = 0
        self.images_train = 0
        self.reponse_train = 0

    def reset(self):
        self.dataset = ""
        self.subject = ""
        self.roi = 1

    def stimuli(self):
        """ returns stimuli to requested dataset"""

        assert self.dataset in ["Trn", "Val"], [
            "Dataset name is not valid, has to be 'Trn' (Train), or 'Val' (Validation)"]
        stimuli = (loadmat('Stimuli.mat'))
        return(stimuli["stim%s" % (self.dataset)])

    def response(self):
        """ returns response to requseted dataset"""
        """ https://courses.cs.ut.ee/MTAT.03.291/2014_spring/uploads/Main/Image%20reconstruction%20from%20fMRI%20data.html"""
        assert self.dataset in ["Trn", "Val"], [
            "Dataset name is not valid, has to be 'Trn' (Train), or 'Val' (Validation)"]
        f = h5py.File('EstimatedResponses.mat')
        roi_subject = f['roi%s' % (self.subject)]

        response = f['data%s%s' % (self.dataset, self.subject)]
        indexes = np.where(roi_subject[0] == self.roi)
        response = np.take(response, indexes, axis=1)
        # don't remove nan, but change to 0 ?  whats better man?
        response = response[:, ~np.isnan(response).any(0)]
        # TODO - write test for filtering voxels  according to region

        return response

    def train_data_set(self):
        self.dataset = "Trn"
        self.images_train = self.stimuli()
        image_size = self.images_train.shape[1]
        ind = random.sample(xrange(self.images_train.shape[0]), 100)
        self.indexes = ind
        self.images_train = np.reshape(
            self.images_train, (self.images_train.shape[0], image_size * image_size))
        self.response_train = self.response()
        # Update images by removing test set
        images = np.delete(self.images_train, self.indexes, axis=0)
        response = np.delete(self.response_train, self.indexes, axis=0)
        return images, response

    def test_data_set(self):
        # Create Testset
        # random returns duplicate indexes,  wanted in test set

        images_test = self.images_train[self.indexes, :]
        response_test = self.response_train[self.indexes, :]
        return images_test, response_test

    def validation_data_set(self):
        self.dataset = "Val"
        images_val = self.stimuli()
        images_val = np.reshape( # val [120,128,128] -> [120,128*128]
            images_val, (images_val.shape[0], images_val.shape[1] * images_val.shape[1]))
        response_val = self.response()
        return images_val, response_val


def test_trainset_shape():
    data = Unwrapdata(subject = "S1", roi = 1)
    stim, resp = data.train_data_set()
    test, test_resp= data.test_data_set()
    val, val_resp = data.validation_data_set()
    assert stim.shape[0] == 1650
    assert resp.shape[0] == 1650
    assert test.shape[0] == 100
    assert val.shape[0] == 120
    assert val_resp.shape[0] == 120
    assert test_resp.shape[0] == 100



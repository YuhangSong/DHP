#!/usr/bin/env python
#
# File Name : cc.py
#
# Description : Computes CC metric #

# Author : Ming Jiang

import numpy as np
import scipy.ndimage


def calc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

class CC():
    '''
    Class for computing CC score for saliency maps
    '''
    def __init__(self,saliconRes):
        self.saliconRes = saliconRes
        self.imgs = self.saliconRes.imgs

    def calc_score(self, gtsAnn, resAnn):
        """
        Computer CC score. A simple implementation
        :param gtsAnn : ground-truth fixation map
        :param resAnn : predicted saliency map
        :return score: int : score
        """

        fixationMap = gtsAnn - np.mean(gtsAnn)
        if np.max(fixationMap) > 0:
            fixationMap = fixationMap / np.std(fixationMap)
        salMap = resAnn - np.mean(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.std(salMap)

        return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

    def compute_score(self, gts, res):
        """
        Computes CC score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values
        :returns: average_score: float (mean CC score computed by averaging scores for all the images)
        """

        assert(gts.keys() == res.keys())
        imgIds = res.keys()
        score = []
        for id in imgIds:
            img = self.imgs[id]
            fixations  = gts[id]
            gtsAnn = {}
            gtsAnn['image_id'] = id
            gtsAnn['fixations'] = fixations
            fixationmap = self.saliconRes.buildFixMap([gtsAnn])
            height,width = (img['height'],img['width'])
            salMap = self.saliconRes.decodeImage(res[id])
            mapheight,mapwidth = np.shape(salMap)
            salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
            score.append(self.calc_score(fixationmap,salMap))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "CC"



if __name__=="__main__":
    nss = CC()
    #more tests here

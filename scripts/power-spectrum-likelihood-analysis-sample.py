#!/usr/bin/env python
# coding: utf-8

import numpy as np
from nbodykit.lab import cosmology
import zeus

import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.pk_tools import pk_tools


def get_fit_selection(kbins, kmin = 0.0, kmax = 0.4, pole_selection = [True, True, True, True, True]):
    k_fit_selection = np.logical_and(kmin<=kbins,kbins<=kmax)
    pole_fit_selection = np.repeat(pole_selection, len(kbins)/len(pole_selection))
    fit_selection = k_fit_selection * pole_fit_selection

    return fit_selection


class PowerSpectrumLikelihood():

    def __init__(self, nmocks):
        datapath = "../data/BOSS_DR12_NGC_z1/"
        datafile = datapath + "ps1D_BOSS_DR12_NGC_z1_COMPnbar_TSC_700_700_700_400_renorm.dat"
        self.W = pk_tools.read_matrix(datapath + "W_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix")
        self.M = pk_tools.read_matrix(datapath + "M_BOSS_DR12_NGC_z1_V6C_1_1_1_1_1_2000_1200.matrix")

        pk_data_dict = pk_tools.read_power(datafile, combine_bins=10)
        kbins, self.pk_data_vector = pk_tools.dict_to_vec(pk_data_dict)

        pk_model_dict = pk_tools.read_power(datafile, combine_bins=1)
        self.krange = pk_model_dict["k"]

        self.z_eff=0.38

        pole_selection=[True, False, True, False, True]
        kmin=0.0
        kmax=0.1
        self.fit_selection = get_fit_selection(kbins, kmin=kmin, kmax=kmax, pole_selection=pole_selection)

        p = np.sum(self.fit_selection)
        Hartlap_factor = (nmocks-p-2) / (nmocks-1)

        cov = pk_tools.read_matrix(datapath + "BOSS_DR12_NGC_z1_cov/BOSS_DR12_NGC_z1_cov_30_30_sample_" + str(nmocks) + ".matrix")
        self.Cinv = Hartlap_factor * np.linalg.inv(cov)


    def pk_model(self, theta):
        h = theta[0]
        b = theta[1]
        f = theta[2]

        if b<0.5 or b>3.5:
            return None

        if f<0 or f>2:
            return None

        try:
            cosmo = cosmology.Planck15.clone(h=h)
            pk_matter = cosmology.LinearPower(cosmo, redshift=self.z_eff, transfer="CLASS")
            pk_matter_k = pk_matter(self.krange)
        except ValueError:
            return None

        P0 = (b**2 + 2/3*b*f + 1/5*f**2) * pk_matter_k
        P2 = (4/3*b*f + 4/7*f**2) * pk_matter_k
        P4 = 8/35*f**2*pk_matter_k

        return np.concatenate((P0, P2, P4))


    def loglike(self, theta):
        pk_model_vector = self.pk_model(theta)
        if pk_model_vector is None:
            return -np.inf

        convolved_model = self.W@self.M@pk_model_vector
        diff = self.pk_data_vector - convolved_model
        fit_diff = diff[self.fit_selection]

        return -0.5 * (fit_diff.T@self.Cinv@fit_diff)


if __name__ == "__main__":
    nmocks = int(sys.argv[1])

    ndim = 3
    nwalkers = 6
    nsteps = 1000

    start = np.zeros((nwalkers, ndim))
    start[:,0] = np.random.uniform(0.33, 0.9, nwalkers)
    start[:,1] = np.random.uniform(0.6, 3.4, nwalkers)
    start[:,2] = np.random.uniform(0.1, 1.9, nwalkers)

    powerlike = PowerSpectrumLikelihood(nmocks=nmocks)

    sampler = zeus.EnsembleSampler(nwalkers, ndim, powerlike.loglike)
    sampler.run_mcmc(start, nsteps, callbacks=zeus.callbacks.SaveProgressCallback("../output/sample_" + str(nmocks) + "_chain.h5", ncheck=50))

    chain = sampler.get_chain(flat=False)
    np.save("../output/sample_" + str(nmocks) + "_chain.npy", chain)

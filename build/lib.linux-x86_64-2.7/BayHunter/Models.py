# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
import copy
import math

class Model(object):
    """Handle interpolating methods for a single model vector."""

    @staticmethod
    def split_modelparams(model):
        model = model[~np.isnan(model)]
        n = int(model.size / 3)  # layers

        vs = model[:n]
        ra = model[n:-n]
        z_vnoi = model[-n:]

        return n, vs, ra, z_vnoi

    @staticmethod
    def get_vp(vs, vpvs=1.73, mantle=[4.3, 1.8]):
        """Return vp from vs, based on crustal and mantle vpvs."""
        ind_m = np.where((vs >= mantle[0]))[0]  # mantle

        vp = vs * vpvs  # correct for crust
        if len(ind_m) == 0:
            return vp
        else:
            ind_m[0] == np.int
            vp[ind_m[0]:] = vs[ind_m[0]:] * mantle[1]
        return vp

    @staticmethod 
    def get_vsv(vs, ra, methods=None):
        if methods == None or methods== 1:
            vsv = vs - 0.5 * ra * 0.01 * vs
        elif methods == 2:
            vsv = math.sqrt(np.square(vs)*(1+ra* 0.01))
        elif methods == 3:
            e = (ra +100) * 0.01
            vsv = math.sqrt(3*(np.square(vs))/(2 + e))
        return vsv 

    @staticmethod 
    def get_vsh(vs, ra, methods=None):
        if methods == None or methods== 1:
            vsh = vs + 0.5 * ra * 0.01 * vs
        elif methods == 2:
            vsh = math.sqrt(np.square(vs) * (1- ra * 0.01))
        elif methods == 3:
            e = (ra +100) * 0.01
            vsh = math.sqrt(3*(np.square(vs))/(1+2/e))
        return vsh         

    @staticmethod 
    def get_vsv_vsh(vs, ra, vs_type=None, methods=None):
        layers = len(vs)
        if vs_type == 'vsv' or vs_type == 2: #Rayleigh wave
            vsv = np.zeros(layers)
            for i in range(layers):
                vsv[i] = Model.get_vsv(vs[i], ra[i], methods)
            return vsv
        elif vs_type == 'vsh'or vs_type == 1: #Love wave  
            vsh = np.zeros(layers)
            for i in range(layers):
                vsh[i] = Model.get_vsh(vs[i], ra[i], methods)
            return vsh
            
    @staticmethod 
    def get_avg_vs(vsv, vsh, methods=None):
        if methods == None or methods== 1:
            vs = 0.5 * (vsv + vsh)
        elif methods == 2:
            vs = 0.5 * (np.square(vsv) + np.square(vsh))
        elif methods == 3:
            vs = math.sqrt((2*np.square(vsv) + np.square(vsh))/3)     
        return vs
               
    @staticmethod 
    def get_vsv_vsh_hist(vs, ra, methods=None):
        """one model or multiple models"""
        if np.ndim(vs) == 1: ###only one model
            vsv = Model.get_vsv_vsh(vs, ra, vs_type='vsv', methods=methods)
            vsh = Model.get_vsv_vsh(vs, ra, vs_type='vsh', methods=methods)
        else:  ###only mutiple model
            vsv = np.empty(vs.shape)
            vsh = np.empty(vs.shape)
            for i in range(np.ndim(vs)):
                vsv[i] = Model.get_vsv_vsh(vs[i], ra[i], vs_type='vsv', methods=methods)
                vsh[i] = Model.get_vsv_vsh(vs[i], ra[i], vs_type='vsh', methods=methods) 
        return vsv, vsh        
 
    @staticmethod
    def get_vp_vs_h(model, vpvs=1.73, mantle=None):
        """Return vp, vs and h from a input model [vs, z_vnoi]"""
        n, vs, ra, z_vnoi = Model.split_modelparams(model)
        # discontinuities:
        z_disc = (z_vnoi[:n-1] + z_vnoi[1:n]) / 2.
        h_lay = (z_disc - np.concatenate(([0], z_disc[:-1])))
        h = np.concatenate((h_lay, [0]))

        if mantle is not None:
            vp = Model.get_vp(vs, vpvs, mantle)
        else:
            vp = vs * vpvs
        return vp, vs, ra, h

    @staticmethod
    def get_stepmodel(model, vpvs=1.73, mantle=None):
        """Return a steplike model from input model, for plotting."""
        vp, vs, ra, h = Model.get_vp_vs_h(model, vpvs, mantle)

        dep = np.cumsum(h)

        # insert steps into velocity model
        dep = np.concatenate([(d, d) for d in dep])
        dep_step = np.concatenate([[0], dep[:-1]])
        vp_step = np.concatenate([(v, v) for v in vp])
        vs_step = np.concatenate([(v, v) for v in vs])
        ra_step = np.concatenate([(r, r) for r in ra])

        dep_step[-1] = np.max([150, dep_step[-1] * 2.5])  # half space

        return vp_step, vs_step, ra_step, dep_step

    @staticmethod
    def get_stepmodel_from_h(h, vs, vpvs=1.73, dep=None, vp=None, ra=None, mantle=None):
        """Return a steplike model from input model."""
        # insert steps into velocity model
        if dep is None:
            dep = np.cumsum(h)

        if vp is None:
            if mantle is not None:
                vp = Model.get_vp(vs, vpvs, mantle)
            else:
                vp = vs * vpvs

        dep = np.concatenate([(d, d) for d in dep])
        dep_step = np.concatenate([[0], dep[:-1]])
        vp_step = np.concatenate([(v, v) for v in vp])
        vs_step = np.concatenate([(v, v) for v in vs])
        ra_step = np.concatenate([(r, r) for r in ra])
	
        dep_step[-1] = dep_step[-1] * 2.5  # half space

        return vp_step, vs_step, ra_step, dep_step

    @staticmethod
    def get_interpmodel(model, dep_int, vpvs=1.73, mantle=None):
        """
        Return an interpolated stepmodel, for (histogram) plotting.

        Model is a vector of the parameters.
        """
        vp_step, vs_step, ra_step, dep_step = Model.get_stepmodel(model, vpvs, mantle)
        vs_int = np.interp(dep_int, dep_step, vs_step)
        vp_int = np.interp(dep_int, dep_step, vp_step)
        ra_int = np.interp(dep_int, dep_step, ra_step)

        return vp_int, vs_int, ra_int


class ModelMatrix(object):
    """
    Handle interpolating methods for a collection of single models.

    Same as the Model class, but for a matrix. Only for Plotting
    or after inversion.
    """

    @staticmethod
    def _delete_nanmodels(models):
        """Remove nan models from model-matrix."""
        cmodels = copy.copy(models)
        mean = np.nanmean(cmodels, axis=1)
        nanidx = np.where((np.isnan(mean)))[0]

        if nanidx.size == 0:
            return cmodels
        else:
            return np.delete(cmodels, nanidx, axis=0)

    @staticmethod
    def _replace_zvnoi_h(models):
        """
        Return model matrix with (vs, h) - models.

        Each model in the matrix is parametrized with (vs, z_vnoi).
        For plotting, h will be computed from z_vnoi."""
        models = ModelMatrix._delete_nanmodels(models)

        for i, model in enumerate(models):
            _, vs, ra, h = Model.get_vp_vs_h(model)
            newmodel = np.concatenate((vs, ra, h))
            models[i][:newmodel.size] = newmodel
        return models

    @staticmethod
    def get_interpmodels(models, dep_int):
        """Return model matrix with interpolated stepmodels.

        Each model in the matrix is parametrized with (vs, z_vnoi)."""
        models = ModelMatrix._delete_nanmodels(models)

        deps_int = np.repeat([dep_int], len(models), axis=0)
        vss_int = np.empty((len(models), dep_int.size))
        ras_int = np.empty((len(models), dep_int.size))

        for i, model in enumerate(models):
            # for vs, dep 2D histogram
            _, vs_int, ra_int = Model.get_interpmodel(model, dep_int)
            vss_int[i] = vs_int
            ras_int[i] = ra_int

        return vss_int, ras_int, deps_int
        
    @staticmethod     
    def get_vsv_vsh_interpmodels(models, dep_int, methods):
        """Return model matrix with interpolated stepmodels.

        Each model in the matrix is parametrized with (vs, z_vnoi)."""
        models = ModelMatrix._delete_nanmodels(models)

        deps_int = np.repeat([dep_int], len(models), axis=0)
        vss_int = np.empty((len(models), dep_int.size))
        vssv_int = np.empty((len(models), dep_int.size))
        vssh_int = np.empty((len(models), dep_int.size))
        ras_int = np.empty((len(models), dep_int.size))

        for i, model in enumerate(models):
            # for vs, dep 2D histogram
            _, vs_int, ra_int = Model.get_interpmodel(model, dep_int)
            vss_int[i] = vs_int
            ras_int[i] = ra_int
            vsv_int, vsh_int = Model.get_vsv_vsh_hist(vs_int, ra_int, methods=methods)
            vssv_int[i] = vsv_int
            vssh_int[i] = vsh_int
            
        return vss_int, vssv_int, vssh_int, ras_int, deps_int

    @staticmethod
    def get_singlemodels(models, dep_int=None, misfits=None, methods=None):
        """Return specific single models from model matrix (vs, depth).
        The model is a step model for plotting.

        -- interpolated
        (1) mean
        (2) median
        (3) minmax
        (4) stdminmax

        -- binned, vs step: 0.025 km/s
                   dep step: 0.5 km or as in dep_int
        (5) mode (histogram)

        -- not interpolated
        (6) bestmisfit   - min misfit
        """
        singlemodels = dict()

        if dep_int is None:
            # interpolate depth to 0.5 km bins.
            dep_int = np.linspace(0, 100, 201)

        #vss_int, ras_int, deps_int = ModelMatrix.get_interpmodels(models, dep_int)
        vss_int, vsv_int, vsh_int,ras_int, deps_int = ModelMatrix.get_vsv_vsh_interpmodels(models, dep_int, methods)
        
        mean=[]
        median=[]
        minmax=[]
        stdminmax=[]
        params_mode=[]
        #for i, params in enumerate([vss_int, ras_int]):
        for i, params in enumerate([vss_int, vsv_int, vsh_int, ras_int, deps_int]):
            # (1) mean, (2) median
            mean.append(np.mean(params, axis=0))
            median.append(np.median(params, axis=0))

            # (3) minmax
            minmax.append(np.array((np.min(params, axis=0), np.max(params, axis=0))).T)

            # (4) stdminmax
            stdmodel = np.std(params, axis=0)
            stdminmodel = mean[i] - stdmodel
            stdmaxmodel = mean[i] + stdmodel

            stdminmax.append(np.array((stdminmodel, stdmaxmodel)).T)

            # (5) mode from histogram
            params_flatten = params.flatten()
            if i == 0:
                interval = 0.025
            if i == 1:
                interval = 0.25
            #paramsbins = int((params_flatten.max() - params_flatten.min()) / interval)
            diff_params_flatten = (params_flatten.max() - params_flatten.min()) 
            # in PlotFromStorage posterior_models2d
            paramsbins = 1 if diff_params_flatten == 0 else int((params_flatten.max() - params_flatten.min()) / interval)
            data = np.histogram2d(params.flatten(), deps_int.flatten(),
                            bins=(paramsbins, dep_int))
            bins, params_bin, dep_bin = np.array(data).T
            params_center = (params_bin[:-1] + params_bin[1:]) / 2.
            dep_center = (dep_bin[:-1] + dep_bin[1:]) / 2.
            params_mode.append(params_center[np.argmax(bins.T, axis=1)])        

        mode = (params_mode[0], params_mode[1], params_mode[2],params_mode[3],dep_center) 
        #mode = (params_mode[0], params_mode[1]) 

        # (6) bestmisfit - min misfit
        if misfits is not None:
            ind = np.argmin(misfits)
            _, vs_best, ra_best, dep_best = Model.get_stepmodel(models[ind])

            singlemodels['minmisfit'] = (vs_best, ra_best, dep_best)
	
        # add models to dictionary
        singlemodels['mean'] = (mean[0], mean[1], mean[2], mean[3], dep_int)
        singlemodels['median'] = (median[0], median[1], median[2], median[3], dep_int)
        singlemodels['minmax'] = (minmax[0].T, minmax[1].T, minmax[2].T, minmax[3].T, dep_int)
        singlemodels['stdminmax'] = (stdminmax[0].T, stdminmax[1].T, stdminmax[2].T, stdminmax[3].T, dep_int)
        singlemodels['mode'] = mode

        return singlemodels

    @staticmethod
    def get_weightedvalues(weights, models=None, likes=None, misfits=None,
                           noiseparams=None, vpvs=None):
        """
        Return weighted matrix of models, misfits and noiseparams, and weighted
        vectors of likelihoods.

        Basically just repeats values, as given by weights.
        """
        weights = np.array(weights, dtype=int)
        wlikes, wmisfits, wmodels, wnoise, wvpvs = (None, None, None, None, None)

        if likes is not None:
            wlikes = np.repeat(likes, weights)

        if misfits is not None:
            if type(misfits[0]) in [int, float, np.float64]:
                wmisfits = np.repeat(misfits, weights)
            else:
                wmisfits = np.ones((np.sum(weights), misfits[0].size)) * np.nan
                n = 0
                for i, misfit in enumerate(misfits):
                    for rep in range(weights[i]):
                        wmisfits[n] = misfit
                        n += 1

        if models is not None:
            wmodels = np.ones((np.sum(weights), models[0].size)) * np.nan

            n = 0
            for i, model in enumerate(models):
                for rep in range(weights[i]):
                    wmodels[n] = model
                    n += 1

        if noiseparams is not None:
            wnoise = np.ones((np.sum(weights), noiseparams[0].size)) * np.nan

            n = 0
            for i, noisepars in enumerate(noiseparams):
                for rep in range(weights[i]):
                    wnoise[n] = noisepars
                    n += 1

        if vpvs is not None:
            wvpvs = np.repeat(vpvs, weights)

        return wmodels, wlikes, wmisfits, wnoise, wvpvs

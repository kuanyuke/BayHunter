# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os
import glob
import logging
import numpy as np
import os.path as op
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict

from BayHunter import utils
from BayHunter import Targets
from BayHunter import Model, ModelMatrix
import matplotlib.colors as colors

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rstate = np.random.RandomState(333)


def vs_round(vs):
    # rounding down to next smaller 0.025 interval
    vs_floor = np.floor(vs)
    return np.round((vs-vs_floor)*40)/40 + vs_floor
    
def ra_round(ra):
    # rounding down to next smaller 0.25 interval
    ra_floor = np.floor(ra)
    return np.round((ra-ra_floor)*4)/4 + ra_floor

def tryexcept(func):
    def wrapper_tryexcept(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as e:
            print('* %s: Plotting was not possible\nErrorMessage: %s'
                  % (func.__name__, e))
            return None
    return wrapper_tryexcept


class PlotFromStorage(object):
    """
    Plot and Save from storage (files).
    No chain object is necessary.

    """
    def __init__(self, configfile):
        condict = self.read_config(configfile)
        self.targets = condict['targets']
        self.ntargets = len(self.targets)
        self.refs = condict['targetrefs'] + ['joint']
        self.priors = condict['priors']
        self.initparams = condict['initparams']

        self.datapath = op.dirname(configfile)
        self.figpath = self.datapath.replace('data', '')
        print('Current data path: %s' % self.datapath)
	
	#self.init_tmpinfo() 
	self.init_tmp = self.initparams.get('temperatures')
        self.ncoldchains = np.count_nonzero(self.init_tmp == 1)  
        self.init_filelists()
        self.init_outlierlist()
        self.get_coldchains()

        self.mantle = self.priors.get('mantle', None)

        self.refmodel = {'model': None,
                         'nlays': None,
                         'noise': None,
                         'vpvs': None}

    def read_config(self, configfile):
        return utils.read_config(configfile)

    def savefig(self, fig, filename):
        if fig is not None:
            outfile = op.join(self.figpath, filename)
            fig.savefig(outfile, bbox_inches="tight")
            plt.close('all')

    def init_outlierlist(self):
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            self.outliers = np.loadtxt(outlierfile, usecols=[0], dtype=int)
            print('Outlier chains from file: %d' % self.outliers.size)
        else:
            print('Outlier chains from file: None')
            self.outliers = np.zeros(0)

    def init_filelists(self):
        filetypes = ['models', 'likes', 'misfits', 'noise', 'vpvs']
        filepattern = op.join(self.datapath, 'tmp???_p%d%s.npy')
        files = []
        size = []

        for ftype in filetypes:
            p1files = sorted(glob.glob(filepattern % (1, ftype)))
            p2files = sorted(glob.glob(filepattern % (2, ftype)))
            files.append([p1files, p2files])
            size.append(len(p1files) + len(p2files))
	
        if len(set(size)) == 1:
            self.modfiles, self.likefiles, self.misfiles, self.noisefiles, \
                self.vpvsfiles = files
        else:
            logger.info('You are missing files. Please check ' +
                        '"%s" for completeness.' % self.datapath)
            logger.info('(filetype, number): ' + str(zip(filetypes, size)))
            
        	
    def get_coldchains(self):
	#temperature        
	init_tmp = self.initparams.get('temperatures')
	self.init_tmp =np.array(init_tmp)

	self.ncoldchains = np.count_nonzero(self.init_tmp == 1)
	self.ngroups = self.initparams.get('ngroups')
	self.pernchains=  len(self.init_tmp)
	for i in range(self.ngroups):
		std = np.array(range(self.ncoldchains)) #standard
		if i == 0:
			self.coldchainsidx=std
		else:
			stdnew =  std + self.pernchains * i
			self.coldchainsidx = np.concatenate((self.coldchainsidx,stdnew ))
           
        
        
    def get_outliers(self, dev):
        """Detect outlier chains.

        The median likelihood from each chain (main phase) is computed.
        Relatively to the most converged chain, outliers are declared.
        Chains with a deviation of likelihood of dev % are declared outliers.

        Chose dev based on actual results.
        """
        nchains = len(self.likefiles[1])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians = np.zeros(nchains) * np.nan
        self.hotchainidxs = []
        self.coldchainidxs = []

        for i, likefile in enumerate(self.likefiles[1]):
            
            cidx, _, _ = self._return_c_p_t(likefile)
            chainlikes = np.load(likefile)
            chainmedian = np.median(chainlikes)
	    
	    if cidx not in self.coldchainsidx:
	    	self.hotchainidxs.append(cidx)
	    else:
            	chainidxs[i] = cidx
            	chainmedians[i] = chainmedian
            	
      	chainidxs = chainidxs[~np.isnan(chainidxs)]
	chainmedians = chainmedians[~np.isnan(chainmedians)]
	
        maxlike = np.max(chainmedians)  # best chain average
        # scores must be smaller 1
        if maxlike > 0:
            scores = chainmedians / maxlike
        elif maxlike < 0:
            scores = maxlike / chainmedians

        outliers = chainidxs[np.where(((1-scores) > dev))]
        outscores = 1 - scores[np.where(((1-scores) > dev))]
        outliers = np.concatenate((outliers, self.hotchainidxs))
        outscores = np.concatenate((outliers, np.zeros(len(self.hotchainidxs)) * np.nan))
        

        if len(outliers) > 0:
            print('Outlier chains found with following chainindices:\n')
            print(outliers)
            outlierfile = op.join(self.datapath, 'outliers.dat')
            with open(outlierfile, 'w') as f:
                f.write('# Outlier chainindices with %.3f deviation condition\n' % dev)
                for i, outlier in enumerate(outliers):
                    f.write('%d\t%.3f\n' % (outlier, outscores[i]))

        return outliers

    def _get_chaininfo(self):
        nmodels = [len(np.load(file)) for file in self.likefiles[1]]
        chainlist = [self._return_c_p_t(file)[0] for file in self.likefiles[1]]
        return chainlist, nmodels

    def save_final_distribution(self, maxmodels=200000, dev=0.05):
        """
        Save the final models from all chains, phase 2.

        As input, all the chain files in self.datapath are used.
        Outlier chains will be detected automatically using % dev. The outlier
        detection is based on the maximum reached (median) likelihood
        by the chains. The other chains are compared to the "best" chain and
        sorted out, if the likelihood deviates more than dev * 100 %.

        > Chose dev based on actual results.

        Maxmodels is the maximum number of models to be saved (.npy).
        The chainmodels are combined to one final distribution file,
        while all models are evenly thinned.
        """

        def save_finalmodels(models, likes, misfits, noise, vpvs):
            """Save chainmodels as pkl file"""
            names = ['models', 'likes', 'misfits', 'noise', 'vpvs']
            print('> Saving posterior distribution.')
            for i, data in enumerate([models, likes, misfits, noise, vpvs]):
                outfile = op.join(self.datapath, 't_%s' % names[i])
                np.save(outfile, data)
                print(outfile)

        # delete old outlier file if evaluating outliers newly
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            os.remove(outlierfile)

        self.outliers = self.get_outliers(dev=dev)

        # due to the forced acceptance rate, each chain should have accepted
        # a similar amount of models. Therefore, a constant number of models
        # will be considered from each chain (excluding outlier chains), to
        # add up to a collection of maxmodels models.
        nchains = int(len(self.likefiles[1]) - self.outliers.size)
        maxmodels = int(maxmodels)
        mpc = int(maxmodels / nchains)  # models per chain

        # # open matrixes and vectors
        allmisfits = None
        allmodels = None
        alllikes = np.ones(maxmodels) * np.nan
        allnoise = np.ones((maxmodels, self.ntargets*2)) * np.nan
        allvpvs = np.ones(maxmodels) * np.nan

        start = 0
        chainidxs, nmodels = self._get_chaininfo()

        for i, cidx in enumerate(chainidxs):
            if cidx in self.outliers:
                continue

            index = np.arange(nmodels[i]).astype(int)
            if nmodels[i] > mpc:
                index = rstate.choice(index, mpc, replace=False)
                index.sort()

            chainfiles = [self.modfiles[1][i], self.misfiles[1][i],
                          self.likefiles[1][i], self.noisefiles[1][i],
                          self.vpvsfiles[1][i]]

            for c, chainfile in enumerate(chainfiles):
                _, _, ftype = self._return_c_p_t(chainfile)
                data = np.load(chainfile)[index]

                if c == 0:
                    end = start + len(data)

                if ftype == 'likes':
                    alllikes[start:end] = data

                elif ftype == 'models':
                    if allmodels is None:
                        allmodels = np.ones((maxmodels, data[0].size)) * np.nan

                    allmodels[start:end, :] = data

                elif ftype == 'misfits':
                    if allmisfits is None:
                        allmisfits = np.ones((maxmodels, data[0].size)) * np.nan

                    allmisfits[start:end, :] = data

                elif ftype == 'noise':
                    allnoise[start:end, :] = data

                elif ftype == 'vpvs':
                    allvpvs[start:end] = data

            start = end

        # exclude nans
        allmodels = allmodels[~np.isnan(alllikes)]
        allmisfits = allmisfits[~np.isnan(alllikes)]
        allnoise = allnoise[~np.isnan(alllikes)]
        allvpvs = allvpvs[~np.isnan(alllikes)]
        alllikes = alllikes[~np.isnan(alllikes)]

        save_finalmodels(allmodels, alllikes, allmisfits, allnoise, allvpvs)

    def _unique_legend(self, handles, labels):
        # if a key is double, the last handle in the row is returned to the key
        legend = OrderedDict(zip(labels, handles))
        return legend.values(), legend.keys()

    def _return_c_p_t(self, filename):
        """Return chainindex, phase number, type of file from filename.
        Only for single chain results.
        """
        c, pt = op.basename(filename).split('.npy')[0].split('_')
        cidx = int(c[3:])
        phase, ftype = pt[:2], pt[2:]

        return cidx, phase, ftype

    def _sort(self, chainidxstring):
        chainidx = int(chainidxstring[1:])
        return chainidx

    def _get_layers(self, models):
        layernumber = np.array([(len(model[~np.isnan(model)]) / 2 - 1)
                                for model in models])
        return layernumber

    @tryexcept
    def plot_refmodel(self, fig, mtype='model', plot_ra = False,**kwargs):
        if fig is not None and self.refmodel[mtype] is not None:
            if mtype == 'nlays':
                nlays = self.refmodel[mtype]
                fig.axes[0].axvline(nlays, color='red', lw=0.5, alpha=0.7)

            if mtype == 'model':
                dep, vs, ra = self.refmodel['model']
                fig.axes[0].plot(vs, dep,**kwargs)
                if len(fig.axes) > 1 and plot_ra:
                    fig.axes[len(fig.axes)-1].plot(ra, dep, **kwargs)

                    if len(fig.axes) == 3:
                        deps = np.unique(dep)
                        for d in deps:
                            fig.axes[1].axhline(d, **kwargs)
                       
                    deps = np.unique(dep)

                elif len(fig.axes) == 2:
                    deps = np.unique(dep)
                    for d in deps:
                        fig.axes[1].axhline(d, **kwargs)

            if mtype == 'noise':
                noise = self.refmodel[mtype]
                for i in range(len(noise)):
                    fig.axes[i].axvline(
                        noise[i], color='red', lw=0.5, alpha=0.7)

            if mtype == 'vpvs':
                vpvs = self.refmodel[mtype]
                fig.axes[0].axvline(vpvs, color='red', lw=0.5, alpha=0.7)
        return fig

# Plot values per iteration.

    def _plot_iitervalues(self, files, ax, layer=0, misfit=0, noise=0, ind=-1):
        unifiles = set([f.replace('p1', 'p2') for f in files])
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, len(unifiles)))
	
        xmin = -self.initparams['iter_burnin']
        xmax = self.initparams['iter_main']

        files.sort()
        n = 0
        for i, file in enumerate(files):
            phase = int(op.basename(file).split('_p')[1][0])
            alpha = (0.4 if phase is 1 else 0.7)
            ls = ('-' if phase is 1 else '-')
            lw = (0.5 if phase is 1 else 0.8)
            chainidx, _, _ = self._return_c_p_t(file)
            color = color_list[n]
	    
            data = np.load(file)
            if layer:
                data = self._get_layers(data)
            if misfit or noise:
                data = data.T[ind]            
        
            iters = (np.linspace(xmin, 0, data.size) if phase is 1 else
                     np.linspace(0, xmax, data.size))
            tmpidx  =  chainidx % self.pernchains
            tmp = self.init_tmp[tmpidx]
            label = 'c%d T=%s' % (chainidx,tmp)
            
            #if chainidx in self.coldchainidxs:
	    #	color ='blue'
	    #else:
	    #	color ='red'
	    	
            ax.plot(iters, data, color=color,
                    ls=ls, lw=lw, alpha=alpha,
                    label=label if phase is 2 else '')
		
            if phase == 2:
                if n == 0:
                    datamax = data.max()
                    datamin = data.min()
                else:
                    datamax = np.max([datamax, data.max()])
                    datamin = np.min([datamin, data.min()])
                n += 1

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(datamin*0.95, datamax*1.05)
        ax.axvline(0, color='k', ls=':', alpha=0.7)

        (abs(xmin) + xmax)
        center = np.array([abs(xmin/2.), abs(xmin) + xmax/2.]) / (abs(xmin) + xmax)
        for i, text in enumerate(['Burn-in phase', 'Exploration phase']):
            ax.text(center[i], 0.97, text,
                    fontsize=12, color='k',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.set_xlabel('# Iteration')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

    @tryexcept
    def plot_iitermisfits(self, nchains=6, ind=-1):
        files = self.misfiles[0][:nchains] + self.misfiles[1][:nchains]#+\
        	#self.misfiles[0][self.ncoldchains:] +  self.misfiles[1][self.ncoldchains:]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, misfit=True, ind=ind)
        ax.set_ylabel('%s misfit' % self.refs[ind])
        return fig

    @tryexcept
    def plot_iiterlikes(self, nchains=6):
        files = self.likefiles[0][:nchains] + self.likefiles[1][:nchains]#+\
        	#self.likefiles[0][self.ncoldchains:] +  self.likefiles[1][self.ncoldchains:]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Likelihood')
        return fig

    @tryexcept
    def plot_iiternoise(self, nchains=6, ind=-1):
        """
        nind = noiseindex, meaning:
        0: 'rfnoise_corr'  # should be const, if gauss
        1: 'rfnoise_sigma'
        2: 'swdnoise_corr'  # should be 0
        3: 'swdnoise_sigma'
        # dependent on number and order of targets.
        """
        files = self.noisefiles[0][:nchains] + self.noisefiles[1][:nchains]#+\
        	#self.noisefiles[0][self.ncoldchains:] +  self.noisefiles[1][self.ncoldchains:]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, noise=True, ind=ind)

        parameter = np.concatenate(
            [['correlation (%s)' % ref, '$\sigma$ (%s)' % ref] for ref in self.refs[:-1]])
        ax.set_ylabel(parameter[ind])
        return fig

    @tryexcept
    def plot_iiternlayers(self, nchains=6):
        files = self.modfiles[0][:nchains] + self.modfiles[1][:nchains]#+\
        	#self.modfiles[0][self.ncoldchains:] +  self.modfiles[1][self.ncoldchains:]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, layer=True)
        ax.set_ylabel('Number of layers')
        return fig

    @tryexcept
    def plot_iitervpvs(self, nchains=6):
        files = self.vpvsfiles[0][:nchains] + self.vpvsfiles[1][:nchains]#+\
        	#self.vpvsfiles[0][self.ncoldchains:] +  self.vpvsfiles[1][self.ncoldchains:]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Vp / Vs')
        return fig


# Posterior distributions as 1D histograms for noise and misfits.
# And as 2D histograms / 1D plot for final velocity-depth models
# Considering weighted models.

    @staticmethod
    def _plot_bestmodels(bestmodels, dep_int=None, vs_methods=None, plot_ra=False):
        if plot_ra:
            fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                                    sharey=True, figsize=(4.4, 7))
        else:
            axes=[]
            fig, ax = plt.subplots(figsize=(4.4, 7))#, squeeze=False)
            axes.append(ax)
        
        models = ['mean', 'median', 'stdminmax']
        vsvmodels = ['Vsv mean', 'Vsv median', 'Vsv stdminmax']
        vshmodels = ['Vsh mean', 'Vsh median', 'Vsh stdminmax']
        colors = ['green', 'blue', 'black']
        vsvcolors = ['darkgreen', 'blue', 'lightblue']
        vshcolors = ['lightgreen', 'red', 'indianred']
        ls = ['-', '--', ':']
        lw = [1, 1, 1]

        singlemodels = ModelMatrix.get_singlemodels(bestmodels, dep_int, methods= vs_methods)
	
        for i, model in enumerate(models):
            vs, vsv, vsh, ra, dep = singlemodels[model]
            
            axes[0].plot(vs.T, dep, color=colors[i], label=model,
                    ls=ls[i], lw=lw[i])
                    
            if plot_ra:
                axes[1].plot(ra.T, dep, color=colors[i], label=model,
                    ls=ls[i], lw=lw[i])
        axes[0].invert_yaxis()
        axes[0].set_ylabel('Depth in km')
        axes[0].set_xlabel('$V_S$ in km/s')
        
 	
        if plot_ra:
            axes[1].set_xlabel('Radial Anisotropy in %')  
         
        han, lab = axes[0].get_legend_handles_labels()
        axes[0].legend(han[:-1], lab[:-1], loc=3)
        #legend = OrderedDict(zip(lab, han))
        #handles= legend.values()
        #labels = legend.keys()
        #axes[0].legend(handles[:-1], labels[:-1], loc=3)
        return fig, axes

    @staticmethod
    def _plot_bestmodels_hist(models, dep_int=None, vs_methods=None, plot_ra=False):
        """
        2D histogram with 30 vs cells and 50 depth cells.
        As plot depth is limited to 100 km, each depth cell is a 2 km.

        pinterf is the number of interfaces to be plot (derived from gradient)
        """
        if dep_int is None:
            dep_int = np.linspace(0, 100, 201)  # interppolate depth to 0.5 km.
            # bins for 2d histogram
            depbins = np.linspace(0, 100, 101)  # 1 km bins
        else:
            maxdepth = int(np.ceil(dep_int.max()))
            interp = dep_int[1] - dep_int[0]
            dep_int = np.arange(dep_int[0], dep_int[-1] + interp / 2., interp / 2.)
            depbins = np.arange(0, maxdepth + 2*interp, interp)  # interp km bins
            # nbin = np.arange(0, maxdepth + interp, interp)  # interp km bins

        # get interfaces, #first
        models2 = ModelMatrix._replace_zvnoi_h(models)
        models2 = np.array([model[~np.isnan(model)] for model in models2])
        yinterf = np.array([np.cumsum(model[int(2*model.size/3):-1])
                            for model in models2])
        yinterf = np.concatenate(yinterf)

        vss_int, ras_int, deps_int = ModelMatrix.get_interpmodels(models, dep_int)
        singlemodels = ModelMatrix.get_singlemodels(models, dep_int=depbins)

        vss_flatten = vss_int.flatten()
        vsinterval = 0.025  # km/s, 0.025 is assumption for vs_round
        # vsbins = int((vss_flatten.max() - vss_flatten.min()) / vsinterval)
        vs_histmin = vs_round(vss_flatten.min())-2*vsinterval
        vs_histmax = vs_round(vss_flatten.max())+3*vsinterval
        vsbins = np.arange(vs_histmin, vs_histmax, vsinterval) # some buffer
        
        ras_flatten = ras_int.flatten()
        rainterval = 1.25  # %, 0.25 is assumption for ra_round
        #rabins = int((ras_flatten.max() - ras_flatten.min()) / rainterval)
        ra_histmin = ra_round(ras_flatten.min())-2*rainterval
        ra_histmax = ra_round(ras_flatten.max())+3*rainterval
        rabins = np.arange(ra_histmin, ra_histmax, rainterval) # some buffer
        
        # initiate plot
        if plot_ra:
            fig, axes = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1]},
                                 sharey=True, figsize=(5, 6.5))
        else:
            fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]},
                                 sharey=True, figsize=(5, 6.5))
        fig.subplots_adjust(wspace=0.05)

        vsv_flatten, vsh_flatten = Model.get_vsv_vsh_hist(vss_flatten, ras_flatten, methods= vs_methods)
        vss_flatten = np.concatenate((vsv_flatten, vsh_flatten))
        deps_flatten = np.concatenate((deps_int.flatten(), deps_int.flatten()))    
        data2d, xedges, yedges = np.histogram2d(vss_flatten, deps_flatten,
                                				bins=(vsbins, depbins))

        axes[0].imshow(data2d.T, extent=(xedges[0], xedges[-1],
        							     yedges[0], yedges[-1]),
        			   origin='lower',
        			   vmax=len(models), aspect='auto')
        if plot_ra:
            data2d_ra, xedges_ra, yedges_ra = np.histogram2d(ras_flatten, deps_int.flatten(),
                                                bins=(rabins, depbins))

            axes[2].imshow(data2d_ra.T, extent=(xedges_ra[0], xedges_ra[-1],
                                         yedges_ra[0], yedges_ra[-1]),
                       origin='lower',
                       vmax=len(models), aspect='auto')

        # plot mean / modes
        # colors = ['green', 'white']
        # for c, choice in enumerate(['mean', 'mode']):
        colors = ['white']
        for c, choice in enumerate(['mode']):
            vs, vsv, vsh, ra, dep = singlemodels[choice]
            #vsv, vsh = Model.get_vsv_vsh_hist(vs, ra, methods= vs_methods)
            color = colors[c]
            #axes[0].plot(vsv, dep, vsh, dep, color=color, lw=1, alpha=0.9, label=choice)
            axes[0].plot(vs, dep, color=color, lw=1, alpha=0.9, label=choice)
            if plot_ra:
                axes[2].plot(ra, dep, color=color, lw=1, alpha=0.9, label=choice)

        vs_mode, vsv_mode, vsh_mode, ra_mode, dep_mode = singlemodels['mode']
        axes[0].legend(loc=3)

        # histogram for interfaces
        data = axes[1].hist(yinterf, bins=depbins, orientation='horizontal',
                            color='lightgray', alpha=0.7,
                            edgecolor='k')
        bins, lay_bin, _ = np.array(data).T
        center_lay = (lay_bin[:-1] + lay_bin[1:]) / 2.

        axes[0].set_ylabel('Depth in km')
        axes[0].set_xlabel('$V_S$ in km/s')

        axes[0].invert_yaxis()

        #axes[0].set_title('%d models' % len(models))
        axes[1].set_xticks([])

        if plot_ra:
            axes[2].set_xlabel('Radial Anisotropy in %')
        return fig, axes

    def _get_posterior_data(self, data, final, chainidx=0):
        if final:
            filetempl = op.join(self.datapath, 't_%s.npy')
        else:
            filetempl = op.join(self.datapath, 't%.3d_p2%s.npy' % (chainidx, '%s'))

        outarrays = []
        for dataset in data:
            datafile = filetempl % dataset
            p2data = np.load(datafile)
            outarrays.append(p2data)

        return outarrays

    def _plot_posterior_distribution(self, data, bins, formatter='%.2f', ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5, 3))

        count, bins, _ = ax.hist(data, bins=bins, color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)
        cbins = (bins[:-1] + bins[1:]) / 2.
        mode = cbins[np.argmax(count)]
        median = np.median(data)

        if formatter is not None:
            text = 'median: %s' % formatter % median
            ax.text(0.97, 0.97, text,
                    fontsize=9, color='k',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.axvline(median, color='k', ls=':', lw=1)
        
        # xticks = np.array(ax.get_xticks())
        # ax.set_xticklabels(xticks, fontsize=8)
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    @tryexcept
    def plot_posterior_likes(self, final=True, chainidx=0):
        likes, = self._get_posterior_data(['likes'], final, chainidx)
        bins = 20
        formatter = '%d'

        ax = self._plot_posterior_distribution(likes, bins, formatter)
        ax.set_xlabel('Likelihood')
        return ax.figure

    @tryexcept
    def plot_posterior_misfits(self, final=True, chainidx=0):
        misfits, = self._get_posterior_data(['misfits'], final, chainidx)

        datasets = [misfit for misfit in misfits.T]
        datasets = datasets[:-1]  # excluding joint misfit
        bins = 20
        formatter = '%.2f'

        fig, axes = plt.subplots(1, len(datasets), figsize=(3.5*len(datasets), 3))
        for i, data in enumerate(datasets):
            axes[i] = self._plot_posterior_distribution(data, bins, formatter, ax=axes[i])
            axes[i].set_xlabel('RMS misfit (%s)' % self.refs[i])

        return fig

    @tryexcept
    def plot_posterior_nlayers(self, final=True, chainidx=0):

        models, = self._get_posterior_data(['models'], final, chainidx)

        # get interfaces
        models = np.array([model[~np.isnan(model)] for model in models])
        layers = np.array([(model.size/3 - 1) for model in models])

        bins = np.arange(np.min(layers), np.max(layers)+2)-0.5

        formatter = '%d'
        ax = self._plot_posterior_distribution(layers, bins, formatter)

        xticks = np.arange(layers.min(), layers.max()+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Number of layers')
        return ax.figure

    @tryexcept
    def plot_posterior_vpvs(self, final=True, chainidx=0):
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)
        bins = 20
        formatter = '%.2f'

        ax = self._plot_posterior_distribution(vpvs, bins, formatter)
        ax.set_xlabel('$V_P$ / $V_S$')
        return ax.figure

    @tryexcept
    def plot_posterior_noise(self, final=True, chainidx=0):
        noise, = self._get_posterior_data(['noise'], final, chainidx)
        label = np.concatenate([['correlation (%s)' % ref, '$\sigma$ (%s)' % ref]
                               for ref in self.refs[:-1]])

        pars = int(len(noise.T)/2)
        fig, axes = plt.subplots(pars, 2, figsize=(7, 3*pars))
        fig.subplots_adjust(hspace=0.2)

        for i, data in enumerate(noise.T):
            if self.ntargets > 1:
                ax = axes[int(i/2)][i % 2]
            else:
                ax = axes[i % 2]

            if np.std(data) == 0:  # constant during inversion
                m = np.mean(data)
                bins = [m-1, m-0.1, m+0.1, m+1]
                formatter = None
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
                ax.text(0.5, 0.5, 'constant: %.2f' % m, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_xticks([])
            else:
                bins = 20
                formatter = '%.4f'
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
            ax.set_xlabel(label[i])
        return fig

    @tryexcept
    def plot_posterior_others(self, final=True, chainidx=0):
        likes, = self._get_posterior_data(['likes'], final, chainidx)

        misfits, = self._get_posterior_data(['misfits'], final, chainidx)
        misfits = misfits.T[-1]  # only joint misfit
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        models, = self._get_posterior_data(['models'], final, chainidx)
        models = np.array([model[~np.isnan(model)] for model in models])
        layers = np.array([(model.size/2 - 1) for model in models])
        nbins = np.arange(np.min(layers), np.max(layers)+2)-0.5

        formatters = ['%d', '%.2f', '%.2f', '%d']
        nbins = [20, 20, 20, nbins]
        labels = ['Likelihood', 'Joint misfit', '$V_P$ / $V_S$', 'Number of layers']

        fig, axes = plt.subplots(2, 2, figsize=(7, 6))
        axes = axes.flatten()
        for i, data in enumerate([likes, misfits, vpvs, layers]):
            ax = axes[i]

            if i == 2 and np.std(data) == 0:  # constant vpvs
                m = np.mean(data)
                bins = [m-1, m-0.1, m+0.1, m+1]
                formatter = None
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
                ax.text(0.5, 0.5, 'constant: %.2f' % m, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_xticks([])
            else:
                formatter = formatters[i]
                bins = nbins[i]
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)

                if i == 3:
                    xticks = np.arange(layers.min(), layers.max()+1)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks)

            ax.set_xlabel(labels[i])
        return ax.figure

    @tryexcept
    def plot_posterior_models1d(self, plot_ra = False, final=True, chainidx=0, depint=1):
        """depint is the depth interpolation used for binning. Default=1km."""
        if final:
            ngroups = self.initparams.get('ngroups')
            nchains = ngroups * len(self.init_tmp)- self.outliers.size
            #nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1
	
        models, = self._get_posterior_data(['models'], final, chainidx)
        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        vs_methods = self.priors['vs_methods']
        
        fig, axes = self._plot_bestmodels(models, dep_int, vs_methods, plot_ra)
        
        axes[0].set_ylim(self.priors['z'][::-1])
        axes[0].grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        if plot_ra:
            axes[1].set_ylim(self.priors['z'][::-1])
            axes[1].grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        fig.suptitle('%d models from %d chains' % (len(models), nchains), y= 0.92)
        return fig

    #@tryexcept
    def plot_posterior_models2d(self, plot_ra = False, final=True, chainidx=0, depint=1):
        if final:
            ngroups = self.initparams.get('ngroups')
            nchains = ngroups * len(self.init_tmp)- self.outliers.size
            #nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        models, = self._get_posterior_data(['models'], final, chainidx)

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        vs_methods = self.priors['vs_methods']
        fig, axes = self._plot_bestmodels_hist(models, dep_int, vs_methods, plot_ra)
        #axes[0].set_xlim(self.priors['vs'])
        axes[0].set_ylim(self.priors['z'][::-1])
        if plot_ra:
            axes[2].set_ylim(self.priors['z'][::-1])
        fig.suptitle('%d models from %d chains' % (len(models), nchains), y= 0.92) 
        return fig


# Plot moho depth - crustal vs tradeoff

    @tryexcept
    def plot_moho_crustvel_tradeoff(self, moho=None, mohovs=None, refmodel=None):
        models, vpvs = self._get_posterior_data(['models', 'vpvs'], final=True)

        if moho is None:
            moho = self.priors['z']
        if mohovs is None:
            mohovs = 4.2  # km/s

        mohos = np.zeros(len(models)) * np.nan
        vscrust = np.zeros(len(models)) * np.nan
        vslastlayer = np.zeros(len(models)) * np.nan
        vsjumps = np.zeros(len(models)) * np.nan

        for i, model in enumerate(models):
            thisvpvs = vpvs[i]
            vp, vs, ra,h = Model.get_vp_vs_h(model, thisvpvs, self.mantle)
            # cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
            # ifaces, vs = cdepth[1::2], cvs[::2]   # interfaces, vs
            ifaces = np.cumsum(h)
            vsstep = np.diff(vs)  # velocity change at interfaces
            mohoidxs = np.argwhere((ifaces > moho[0]) & (ifaces < moho[1]))
            if len(mohoidxs) == 0:
                continue

            # mohoidx = mohoidxs[np.argmax(vsstep[mohoidxs])][0]
            mohoidxs = mohoidxs.flatten()

            mohoidxs_vs = np.where((vs > mohovs))[0]-1
            if len(mohoidxs_vs) == 0:
                continue

            mohoidx = np.intersect1d(mohoidxs, mohoidxs_vs)
            if len(mohoidx) == 0:
                continue
            mohoidx = mohoidx[0]
            # ------

            thismoho = ifaces[mohoidx]
            crustmean = np.sum(vs[:(mohoidx+1)] * h[:(mohoidx+1)]) / ifaces[mohoidx]
            lastvs = vs[mohoidx]
            vsjump = vsstep[mohoidx]

            mohos[i] = thismoho
            vscrust[i] = crustmean
            vslastlayer[i] = lastvs
            vsjumps[i] = vsjump

        # exclude nan values
        mohos = mohos[~np.isnan(vsjumps)]
        vscrust = vscrust[~np.isnan(vsjumps)]
        vslastlayer = vslastlayer[~np.isnan(vsjumps)]
        vsjumps = vsjumps[~np.isnan(vsjumps)]

        fig, ax = plt.subplots(2, 4, figsize=(11, 6))
        fig.subplots_adjust(hspace=0.05)
        fig.subplots_adjust(wspace=0.05)

        labels = ['$V_S$ last crustal layer', '$V_S$ crustal mean', '$V_S$ increase']
        bins = 50

        for n, xdata in enumerate([vslastlayer, vscrust, vsjumps]):
            try:
                histdata = ax[0][n].hist(xdata, bins=bins,
                                         color='darkblue', alpha=0.7,
                                         edgecolor='white', linewidth=0.4)

                median = np.median(xdata)
                ax[0][n].axvline(median, color='k', ls='--', lw=1.2, alpha=1)
                stats = 'median:\n%.2f km/s' % median
                ax[0][n].text(0.97, 0.97, stats,
                              fontsize=9, color='k',
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform=ax[0][n].transAxes)
            except:
                pass

        for n, xdata in enumerate([vslastlayer, vscrust, vsjumps]):
            try:
                ax[1][n].set_xlabel(labels[n])

                data = ax[1][n].hist2d(xdata, mohos, bins=bins)
                data2d, xedges, yedges, _ = np.array(data).T

                xi, yi = np.unravel_index(data2d.argmax(), data2d.shape)
                x_mode = ((xedges[:-1] + xedges[1:]) / 2.)[xi]
                y_mode = ((yedges[:-1] + yedges[1:]) / 2.)[yi]

                ax[1][n].axhline(y_mode, color='white', ls='--', lw=0.5, alpha=0.7)
                ax[1][n].axvline(x_mode, color='white', ls='--', lw=0.5, alpha=0.7)

                xmin, xmax = ax[1][n].get_xlim()
                ax[0][n].set_xlim([xmin, xmax])
            except:
                pass

            ax[0][n].set_yticks([])
            ax[0][n].set_yticklabels([], visible=False)
            ax[0][n].set_xticklabels([], visible=False)

        ax[1][1].set_yticklabels([], visible=False)
        ax[1][2].set_yticklabels([], visible=False)
        ax[1][3].set_yticklabels([], visible=False)
        ax[1][0].set_ylabel('Moho depth in km')

        # plot moho 1d histogram
        histdata = ax[1][3].hist(mohos, bins=bins, orientation='horizontal',
                                 color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)

        median = np.median(mohos)
        std = np.std(mohos)
        print('moho: %.4f +- %.4f km' % (median, std))
        ax[1][3].axhline(median, color='k', ls='--', lw=1.2, alpha=1)
        stats = 'median:\n%.2f km' % median
        ax[1][3].text(0.97, 0.97, stats,
                      fontsize=9, color='k',
                      horizontalalignment='right',
                      verticalalignment='top',
                      transform=ax[1][3].transAxes)
        ymin, ymax = ax[1][0].get_ylim()
        # ymin, ymax = median - 4*std, median + 4*std
        ax[1][0].set_ylim(ymin, ymax)
        ax[1][1].set_ylim(ymin, ymax)
        ax[1][2].set_ylim(ymin, ymax)
        ax[1][3].set_ylim(ymin, ymax)

        ax[1][3].set_xticklabels([], visible=False)
        ax[1][3].set_yticks([])
        ax[1][3].set_yticklabels([], visible=False)
        ax[0][3].axis('off')

        if refmodel is not None:
            dep, vs = refmodel
            h = (dep[1:] - dep[:-1])[::2]
            ifaces, lvs = dep[1::2], vs[::2]

            vsstep = np.diff(lvs)  # velocity change at interfaces
            mohoidxs = np.argwhere((ifaces > moho[0]) & (ifaces < moho[1]))
            mohoidx = mohoidxs[np.argmax(vsstep[mohoidxs])][0]
            truemoho = ifaces[mohoidx]
            truecrust = np.sum(lvs[:(mohoidx+1)] * h[:(mohoidx+1)]) / ifaces[mohoidx]
            truevslast = lvs[mohoidx]
            truevsjump = vsstep[mohoidx]

            for n, xdata in enumerate([truevslast, truecrust, truevsjump]):
                ax[1][n].axhline(truemoho, color='red', ls='--', lw=0.5, alpha=0.7)
                ax[1][n].axvline(xdata, color='red', ls='--', lw=0.5, alpha=0.7)

        return fig

# Plot current models and data fits. also plot best data fit incl. model.

    @tryexcept
    def plot_currentmodels(self, nchains, plot_ra = False):
        """Return fig.

        Plots the first nchains chains, no matter of outlier status.
        """
        fig = plt.figure(figsize=(4, 6.5))

        if plot_ra:
            spec = gridspec.GridSpec( ncols=2, nrows=1, width_ratios=[3, 1])
            ax0 = fig.add_subplot(spec[0])
            ax1 = fig.add_subplot(spec[1])
        else:
            ax0 = fig.add_subplot(1,1,1)

        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, nchains))
        vs_methods = self.priors['vs_methods']

        for i, modfile in enumerate(self.modfiles[1][:nchains]):
            chainidx, _, _ = self._return_c_p_t(modfile)
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            currentvpvs = vpvs[-1]
            currentmodel = models[-1]

            color = color_list[i]
            vp, vs, ra, h = Model.get_vp_vs_h(currentmodel, currentvpvs, self.mantle)
            cvp, cvs, cra, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp, ra=ra)

            cvsv = Model.get_vsv_vsh(cvs, cra, vs_type='vsv', methods= vs_methods)
            cvsh = Model.get_vsv_vsh(cvs, cra, vs_type='vsh', methods= vs_methods)

            label = 'c%d / %d' % (chainidx, vs.size-1)
            #ax0.plot(cvsv, cdepth, cvsh, cdepth,color=color, ls='-', lw=0.8,
            #        alpha=0.7, label=label)
            ax0.plot(cvs, cdepth,color=color, ls='-', lw=0.8,
                    alpha=0.7, label=label)
            if plot_ra:
                ax1.plot(cra, cdepth, color=color, ls='-', lw=0.8,
                    alpha=0.7)

        ax0.invert_yaxis()
        ax0.set_xlabel('$V_S$ in km/s')
        ax0.set_ylabel('Depth in km')
        # ax.set_xlim(self.priors['vs'])
        ax0.set_ylim(self.priors['z'][::-1])
        ax0.grid(color='gray', alpha=0.6, ls=':', lw=0.5)

        if plot_ra:    
            #ax1.invert_yaxis()
            ax1.set_xlabel('Radial Anisotropy in %')
            ax1.yaxis.set_visible(False)
            ax1.set_ylim(self.priors['z'][::-1])
            ax1.grid(color='gray', alpha=0.6, ls=':', lw=0.5)

            fig.suptitle('Current models', y= 0.92)
            ax0.legend(loc='center left', bbox_to_anchor=(0, 0.15))
        else:
            ax0.set_title('Current models')
            ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig

    @tryexcept
    def plot_currentdatafits(self, nchains):
        """Plot the first nchains chains, no matter of outlier status.
        """
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, nchains))
        targets = Targets.JointTarget(targets=self.targets)

        fig, ax = targets.plot_obsdata(mod=False)

        for i, modfile in enumerate(self.modfiles[1][:nchains]):
            color = color_list[i]
            chainidx, _, _ = self._return_c_p_t(modfile)
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            currentvpvs = vpvs[-1]
            currentmodel = models[-1]

            vp, vs, ra, h = Model.get_vp_vs_h(currentmodel, currentvpvs, self.mantle)
            rho = vp * 0.32 + 0.77

            jmisfit = 0
            for n, target in enumerate(targets.targets):
                if target.ref == 'prf' or target.ref == 'srf':
                    xmod, ymod = target.moddata.plugin.run_model(
                        h=h, vp=vp, vs=vs, rho=rho)
                else:
                    vs_methods = self.priors['vs_methods']
                    xmod, ymod = target.moddata.plugin.run_model(
                    h=h, vp=vp, vs=vs, ra=ra, rho=rho, methods=vs_methods) 

                yobs = target.obsdata.y
                misfit = target.valuation.get_rms(yobs, ymod)
                jmisfit += misfit
		
                tmpidx  =  chainidx % self.pernchains
                tmp = self.init_tmp[tmpidx]
               
            
                if chainidx in self.coldchainidxs:
                	color ='green'
                else:
                	color ='red'
	    	
                label = ''
                if len(targets.targets) > 1:
                    if ((len(targets.targets) - 1) - n) < 1e-2:
                        label = 'c%d  / tmp = %d / %.3f' % (chainidx, tmp, jmisfit)
                    ax[n].plot(xmod, ymod, color=color, alpha=0.7, lw=0.8,
                               label=label)
                else:
                    label = 'c%d / tmp =%d / %.3f' % (chainidx, tmp, jmisfit)
                    
                    ax.plot(xmod, ymod, color=color, alpha=0.5, lw=0.7,
                            label=label)

        if len(targets.targets) > 1:
            ax[0].set_title('Current data fits')
            idx = len(targets.targets) - 1
            han, lab = ax[idx].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)
        else:
            ax.set_title('Current data fits')
            han, lab = ax.get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax.legend().set_visible(False)

        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig

    @tryexcept
    def plot_bestmodels(self, plot_ra = False):
        """Return fig.

        Plot the best (fit) models ever discovered per each chain,
        ignoring outliers.
        """
        fig = plt.figure(figsize=(4, 6.5))

        if plot_ra:
            spec = gridspec.GridSpec( ncols=2, nrows=1, width_ratios=[3, 1])
            ax0 = fig.add_subplot(spec[0])
            ax1 = fig.add_subplot(spec[1])
        else:
            ax0 = fig.add_subplot(1,1,1)

        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan

        modfiles = self.modfiles[1]
        vs_methods = self.priors['vs_methods']

        for i, modfile in enumerate(modfiles):
            chainidx, _, _ = self._return_c_p_t(modfile)
            if chainidx in self.outliers:
                continue
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            misfits = np.load(modfile.replace('models', 'misfits')).T[-1]
            bestmodel = models[np.argmin(misfits)]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]

            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestmodel = bestmodel
                thebestvpvs = bestvpvs
                thebestchain = chainidx

            vp, vs, ra, h = Model.get_vp_vs_h(bestmodel, bestvpvs, self.mantle)
            cvp, cvs, cra, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp, ra =ra)
            #cvsv = Model.get_vsv_vsh(cvs, cra, vs_type='vsv', methods= vs_methods)
            #cvsh = Model.get_vsv_vsh(cvs, cra, vs_type='vsh', methods= vs_methods)

            #ax0.plot(cvsv, cdepth, cvsh, cdepth, color='k', ls='-', lw=0.8, alpha=0.5)
            ax0.plot(cvs, cdepth, color='k', ls='-', lw=0.8, alpha=0.5)
            if plot_ra:
                ax1.plot(cra, cdepth, color='k', ls='-', lw=0.8, alpha=0.5)

        # label = 'c%d' % thebestchain
        # vp, vs, h = Model.get_vp_vs_h(thebestmodel, thebestvpvs, self.mantle)
        # cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
        # ax.plot(cvs, cdepth, color='red', ls='-', lw=1,
        #         alpha=0.8, label=label)

        ax0.invert_yaxis()
        ax0.set_xlabel('$V_S$ in km/s')
        ax0.set_ylabel('Depth in km')
        # ax.set_xlim(self.priors['vs'])
        ax0.set_ylim(self.priors['z'][::-1])
        ax0.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        # ax.legend(loc=3)

        if plot_ra:
            #ax1.invert_yaxis()
            ax1.yaxis.set_visible(False)
            ax1.set_xlabel('Radial Anisotropy(%)')
            ax1.set_ylim(self.priors['z'][::-1])
            ax1.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
            fig.suptitle('Best fit models from %d chains' %
                     (len(modfiles)-self.outliers.size), y= 0.92)
        else:
            ax0.set_title('Best fit models from %d chains' %
            (len(modfiles)-self.outliers.size)) 
        return fig

    @tryexcept
    def plot_bestdatafits(self):
        """Plot best data fits from each chain and ever best,
        ignoring outliers."""
        targets = Targets.JointTarget(targets=self.targets)
        fig, ax = targets.plot_obsdata(mod=False)
        
        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan

        modfiles = self.modfiles[1]
        for i, modfile in enumerate(modfiles):
            chainidx, _, _ = self._return_c_p_t(modfile)
            if chainidx in self.outliers:
                continue
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            misfits = np.load(modfile.replace('models', 'misfits')).T[-1]
            bestmodel = models[np.argmin(misfits)]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]

            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestmodel = bestmodel
                thebestvpvs = bestvpvs
                thebestchain = chainidx

            vp, vs, ra, h = Model.get_vp_vs_h(bestmodel, bestvpvs, self.mantle)
            rho = vp * 0.32 + 0.77
            
            for n, target in enumerate(targets.targets):
                if target.ref == 'prf' or target.ref == 'srf':
                    xmod, ymod = target.moddata.plugin.run_model(
                        h=h, vp=vp, vs=vs, rho=rho)
                else:
                    vs_methods = self.priors['vs_methods']
                    xmod, ymod = target.moddata.plugin.run_model(
                    h=h, vp=vp, vs=vs, ra=ra, rho=rho, methods=vs_methods) 

                if len(targets.targets) > 1:
                    ax[n].plot(xmod, ymod, color='k', alpha=0.5, lw=0.7)
                else:
                    ax.plot(xmod, ymod, color='k', alpha=0.5, lw=0.7)

        if len(targets.targets) > 1:
            ax[0].set_title('Best data fits from %d chains' %
                            (len(modfiles)-self.outliers.size))
            # idx = len(targets.targets) - 1
            han, lab = ax[0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)
        else:
            ax.set_title('Best data fits from %d chains' %
                         (len(modfiles)-self.outliers.size))
            han, lab = ax.get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax.legend().set_visible(False)

        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig

    @tryexcept
    def plot_finaldatafits(self, final=True, depint=1):
        targets = Targets.JointTarget(targets=self.targets)
        fig, ax = targets.plot_obsdata(mod=False)
        
        ngroups = self.initparams.get('ngroups')
        nchains = ngroups * len(self.init_tmp)- self.outliers.size
	chainidx=0
        bestmodels, = self._get_posterior_data(['models'], final, chainidx)
        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        vs_methods = self.priors['vs_methods']
        
        models = ['mean']#, 'median']
        colors = ['green']#, 'black']
	
        singlemodels = ModelMatrix.get_singlemodels(bestmodels, dep_int, methods= vs_methods)
        posterior_vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)
        mean_vpvs = np.mean(posterior_vpvs, axis=0)
	median_vpvs = np.median(posterior_vpvs, axis=0)
	vpvs = [mean_vpvs, median_vpvs]
        for i, model in enumerate(models):
            vs, vsv, vsh, ra, dep = singlemodels[model]
	    vpvs = vpvs[i]
	    vp = vpvs * vs
            rho = vp * 0.32 + 0.77
            h = np.ones(len(dep)) * depint
            h[-1]=0
            
            for n, target in enumerate(targets.targets):
                if target.ref == 'prf' or target.ref == 'srf':
                    xmod, ymod = target.moddata.plugin.run_model(
                        h=h, vp=vp, vs=vs, rho=rho)
                else:
                    vs_methods = self.priors['vs_methods']
                    xmod, ymod = target.moddata.plugin.run_model(
                    h=h, vp=vp, vs=vs, ra=ra, rho=rho, methods=vs_methods) 

                if len(targets.targets) > 1:
                    ax[n].plot(xmod, ymod, color=colors[i],  lw=1,label =model )
                else:
                    ax.plot(xmod, ymod, color=colors[i], lw=1,label =model)
                       
        if len(targets.targets) > 1:
            ax[0].set_title('Reference data fits')
            han, lab = ax[0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)
        else:
            ax.set_title('Reference data fits')
            han, lab = ax.get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax.legend().set_visible(False)

        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig          
            
            
    @tryexcept
    def plot_rfcorr(self, rf='prf'):
        from BayHunter import SynthObs

        p2models, p2noise, p2misfits, p2vpvs = self._get_posterior_data(
            ['models', 'noise', 'misfits', 'vpvs'], final=True)

        fig, axes = plt.subplots(2, sharex=True, sharey=True)
        ind = self.refs.index(rf)
        best = np.argmin(p2misfits.T[ind])
        model = p2models[best]
        vpvs = p2vpvs[best]

        target = self.targets[ind]
        x, y = target.obsdata.x, target.obsdata.y
        vp, vs, ra, h = Model.get_vp_vs_h(model, vpvs, self.mantle)
        rho = vp * 0.32 + 0.77

        _, ymod = target.moddata.plugin.run_model(
            h=h, vp=vp, vs=vs, rho=rho)
        yobs = target.obsdata.y
        yresiduals = yobs - ymod

        # axes[0].set_title('Residuals [dobs-g(m)] obtained with best fitting model m')
        axes[0].plot(x, yresiduals, color='k', lw=0.7, label='residuals')

        corr, sigma = p2noise[best][2*ind:2*(ind+1)]
        yerr = SynthObs.compute_gaussnoise(y, corr=corr, sigma=sigma)
        # axes[1].set_title('One Realization of random noise from inferred CeRF')
        axes[1].plot(x, yerr, color='k', lw=0.7, label='noise realization')
        axes[1].set_xlabel('Time in s')

        axes[0].legend(loc=4)
        axes[1].legend(loc=4)
        axes[0].grid(color='gray', ls=':', lw=0.5)
        axes[1].grid(color='gray', ls=':', lw=0.5)
        axes[0].set_xlim([x[0], x[-1]])

        return fig

    def merge_pdfs(self):
        from PyPDF2 import PdfFileReader, PdfFileWriter

        outputfile = op.join(self.figpath, 'c_summary.pdf')
        output = PdfFileWriter()
        pdffiles = glob.glob(op.join(self.figpath + os.sep + 'c_*.pdf'))
        pdffiles.sort(key=op.getmtime)

        for pdffile in pdffiles:
            if pdffile == outputfile:
                continue

            document = PdfFileReader(open(pdffile, 'rb'))
            for i in range(document.getNumPages()):
                output.addPage(document.getPage(i))

        with open(outputfile, "wb") as f:
            output.write(f)

    def save_chainplots(self, cidx=0, refmodel=dict(), depint=None):
        """
        Refmodel is a dictionary and must contain plottable values:
        - 'vs' and 'dep' for the vs-depth plots, will be plotted as given
        - 'rfnoise_corr', 'rfnoise_sigma', 'swdnoise_corr', 'swdnoise_sigma' -
        in this order as noise parameter reference in histogram plots
        - 'nlays' number of layers as reference

        Only given values will be plotted.

        - depint is the interpolation only for histogram plotting.
        Default is 1 km. A finer interpolation increases the plotting time.
        """
        self.refmodel.update(refmodel)
        # plot chain specific posterior distributions

        fig5a = self.plot_posterior_misfits(final=False, chainidx=cidx)
        self.savefig(fig5a, 'c%.3d_posterior_misfit.pdf' % cidx)

        fig5b = self.plot_posterior_nlayers(final=False, chainidx=cidx)
        self.plot_refmodel(fig5b, 'nlays')
        self.savefig(fig5b, 'c%.3d_posterior_nlayers.pdf' % cidx)

        fig5c = self.plot_posterior_noise(final=False, chainidx=cidx)
        self.plot_refmodel(fig5c, 'noise')
        self.savefig(fig5c, 'c%.3d_posterior_noise.pdf' % cidx)

        fig5d = self.plot_posterior_models1d(
            final=False, chainidx=cidx, depint=depint)
        self.plot_refmodel(fig5d, 'model', color='k', lw=1)
        self.savefig(fig5d, 'c%.3d_posterior_models1d.pdf' % cidx)

        fig5e = self.plot_posterior_models2d(
            final=False, chainidx=cidx, depint=depint)
        self.plot_refmodel(fig5e, 'model', color='red', lw=0.5, alpha=0.7)
        self.savefig(fig5e, 'c%.3d_posterior_models2d.pdf' % cidx)

    def save_plots(self, nchains=5, refmodel=dict(), plot_ra =False, depint=1):
        """
        Refmodel is a dictionary and must contain plottable values:
        - 'vs' and 'dep' (np.arrays) for the vs-depth plots, will be plotted as given
        - noise parameters, if e.g., inverting for RF and SWD are:
        'rfnoise_corr', 'rfnoise_sigma', 'swdnoise_corr', 'swdnoise_sigma',
        (depends on number of targets, but order must be correlation / sigma)
        - 'nlays' number of layers as reference

        Only given values will be plotted.

        - depint is the interpolation only for histogram plotting.
        Default is 1 km. A finer interpolation increases the plotting time.
        """
        self.refmodel.update(refmodel)

        nchains = np.min([nchains, len(self.likefiles[1])])

        # plot values changing over iteration
        fig1a = self.plot_iiterlikes(nchains=nchains)
        self.savefig(fig1a, 'c_iiter_likes.pdf')

        fig1b = self.plot_iitermisfits(nchains=nchains, ind=-1)
        self.savefig(fig1b, 'c_iiter_misfits.pdf')

        fig1c = self.plot_iiternlayers(nchains=nchains)
        self.savefig(fig1c, 'c_iiter_nlayers.pdf')

        fig1d = self.plot_iitervpvs(nchains=nchains)
        self.savefig(fig1d, 'c_iiter_vpvs.pdf')

        for i in range(self.ntargets):
            ind = i * 2 + 1
            fig1d = self.plot_iiternoise(nchains=nchains, ind=ind)
            self.savefig(fig1d, 'c_iiter_noisepar%d.pdf' % ind)

        # plot current models and datafit
        fig3a = self.plot_currentmodels(nchains=nchains, plot_ra = plot_ra)
        self.plot_refmodel(fig3a, 'model', color='k', lw=1)
        self.savefig(fig3a, 'c_currentmodels.pdf')

        fig3b = self.plot_currentdatafits(nchains=nchains)
        self.savefig(fig3b, 'c_currentdatafits.pdf')

        # plot final posterior distributions
        fig2b = self.plot_posterior_nlayers()
        self.plot_refmodel(fig2b, 'nlays')
        self.savefig(fig2b, 'c_posterior_nlayers.pdf')

        fig2b = self.plot_posterior_vpvs()
        self.plot_refmodel(fig2b, 'vpvs')
        self.savefig(fig2b, 'c_posterior_vpvs.pdf')

        fig2c = self.plot_posterior_noise()
        self.plot_refmodel(fig2c, 'noise')
        self.savefig(fig2c, 'c_posterior_noise.pdf')

        fig2d = self.plot_posterior_models1d(plot_ra = plot_ra, depint=depint)
        self.plot_refmodel(fig2d, 'model', plot_ra = plot_ra, color='k', lw=1)
        self.savefig(fig2d, 'c_posterior_models1d.pdf')
        fig2e = self.plot_posterior_models2d(plot_ra = plot_ra, depint=depint)
        self.plot_refmodel(fig2e, 'model', plot_ra = plot_ra, color='red', lw=0.5, alpha=0.7)
        self.savefig(fig2e, 'c_posterior_models2d.pdf')


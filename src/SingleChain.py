# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import copy
import time
import numpy as np
import os.path as op

from BayHunter import Model, ModelMatrix
from BayHunter import utils

import logging
logger = logging.getLogger()


PAR_MAP = {'vsmod': 0, 'zvmod': 1, 'birth': 2, 'death': 2,
           'noise': 3, 'vpvs': 4, 'ramod': 5,}


class SingleChain(object):

    def __init__(self, targets, chainidx=0, initparams={}, modelpriors={},
                 sharedcurrentlikes=None, sharediiter=None, partmp=None, random_seed=None):
                 
        self.chainidx = chainidx
        self.rstate = np.random.RandomState(random_seed)

        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.initparams.update(initparams)
        self.priors.update(modelpriors)
        self.dv = (self.priors['vs'][1] - self.priors['vs'][0])
        if self.priors['ra'] is None or \
        type(self.priors['ra']) == np.float or type(self.priors['ra'])  == np.int:
        	self.dra = 1
        	self.modifications_ra = False
        else:
        	self.dra = (self.priors['ra'][1] - self.priors['ra'][0])
        	self.modifications_ra = True
        	
	self.init_tmp = self.initparams.get('temperatures')
        self.ngroups = self.initparams.get('ngroups')
        self.nchains = self.ngroups * len(self.init_tmp)
        self.station = self.initparams['station']

        # set targets and inversion specific parameters
        self.targets = targets

        # set parameters
        self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
        self.iterations = self.iter_phase1 + self.iter_phase2
        self.iiter = -self.iter_phase1
        self.lastmoditer = self.iiter
        self.addiiter = 0.5 *self.iterations

        self.propdist = np.array(self.initparams['propdist'])
        self.acceptance = self.initparams['acceptance']
        self.thickmin = self.initparams['thickmin']
        self.maxlayers = int(self.priors['layers'][1]) + 1

        self.lowvelperc = self.initparams['lvz']
        self.highvelperc = self.initparams['hvz']
        self.mantle = self.priors['mantle']
        
	#parallel tempering setting
        self.freq_swap = self.initparams.get('freq_swap')
        self.init_swap = self.initparams.get('init_swap')
	self.partmp = partmp

        # chain models
        self._init_chainarrays(sharedcurrentlikes, sharediiter)

        # init model and values
        self._init_model_and_currentvalues()


# init model and misfit / likelihood

    def _init_model_and_currentvalues(self):
        ivpvs = self.draw_initvpvs()
        self.currentvpvs = ivpvs
        imodel = self.draw_initmodel()
        # self.currentmodel = imodel
        inoise, corrfix = self.draw_initnoiseparams()
        # self.currentnoise = inoise
        vs_methods = self.priors['vs_methods']

        rcond = self.initparams['rcond']
        self.set_target_covariance(corrfix[::2], inoise[::2], rcond)

        vp, vs, ra, h = Model.get_vp_vs_h(imodel, ivpvs, self.mantle)
        self.targets.evaluate(h=h, vp=vp, vs=vs, ra=ra, methods=vs_methods, noise=inoise)

        # self.currentmisfits = self.targets.proposalmisfits
        # self.currentlikelihood = self.targets.proposallikelihood

        logger.debug((vs, h))

        self.n = 0  # accepted models counter
        self.accept_as_currentmodel(imodel, inoise, ivpvs)
        self.append_currentmodel()

    def draw_initmodel(self):
        keys = self.priors.keys()
        zmin, zmax = self.priors['z']
        vsmin, vsmax = self.priors['vs']
        layers = self.priors['layers'][0] + 1  # half space
        #layers = self.priors['layers'][0] + 1  # half space
        #if (self.chainidx % 2 ) == 0:
        #	layers = self.priors['layers'][0] + 2  # half space
        #elif  (self.chainidx % 3 ) == 0:
	#	layers = self.priors['layers'][0] + 2  # half space
        #else:
        #	layers = self.priors['layers'][0] + 1  # half space
        #layers = self.priors['layers'][0] + 1  # half space

        if self.priors['ra'] is None:
            ra = np.zeros(layers)
        elif type(self.priors['ra']) == np.float or type(self.priors['ra'])  == np.int:
            ra = np.ones(layers) * self.priors['ra']
        else:
            ramin, ramax = self.priors['ra'] 
            ra = self.rstate.uniform(low=ramin, high=ramax, size=layers)
        ra.sort() 
          
        vs = self.rstate.uniform(low=vsmin, high=vsmax, size=layers)
        vs.sort()

        if (self.priors['mohoest'] is not None and layers > 1):
            mean, std = self.priors['mohoest']
            moho = self.rstate.normal(loc=mean, scale=std)
            tmp_z = self.rstate.uniform(1, np.min([5, moho]))  # 1-5
            tmp_z_vnoi = [moho-tmp_z, moho+tmp_z]

            if (layers - 2) == 0:
                z_vnoi = tmp_z_vnoi
            else:
                z_vnoi = np.concatenate((
                    tmp_z_vnoi,
                    self.rstate.uniform(low=zmin, high=zmax, size=(layers - 2))))

        else:  # no moho estimate
            z_vnoi = self.rstate.uniform(low=zmin, high=zmax, size=layers)

        z_vnoi.sort()
        model = np.concatenate((vs, ra, z_vnoi))

        return(model if self._validmodel(model)
               else self.draw_initmodel())

    def draw_initnoiseparams(self):
        # for each target the noiseparams are (corr and sigma)
        noiserefs = ['noise_corr', 'noise_sigma']
        init_noise = np.ones(len(self.targets.targets)*2) * np.nan
        corrfix = np.zeros(len(self.targets.targets)*2, dtype=bool)

        self.noisepriors = []
        for i, target in enumerate(self.targets.targets):
            for j, noiseref in enumerate(noiserefs):
                idx = (2*i)+j
                noiseprior = self.priors[target.noiseref + noiseref]

                if type(noiseprior) in [int, float, np.float64]:
                    corrfix[idx] = True
                    init_noise[idx] = noiseprior
                else:
                    init_noise[idx] = self.rstate.uniform(
                        low=noiseprior[0], high=noiseprior[1])

                self.noisepriors.append(noiseprior)

        self.noiseinds = np.where(corrfix == 0)[0]
        if len(self.noiseinds) == 0:
            logger.warning('All your noise parameters are fixed. On Purpose?')

        return init_noise, corrfix

    def draw_initvpvs(self):
        if type(self.priors['vpvs']) == np.float:
            return self.priors['vpvs']

        vpvsmin, vpvsmax = self.priors['vpvs']
        return self.rstate.uniform(low=vpvsmin, high=vpvsmax)

    def set_target_covariance(self, corrfix, noise_corr, rcond=None):
        # SWD noise hyper-parameters: if corr is not 0, the correlation of data
        # points assumed will be exponential.
        # RF noise hyper-parameters: if corr is not 0, but fixed, the
        # correlation between data points will be assumed gaussian (realistic).
        # if the prior for RFcorr is a range, the computation switches
        # to exponential correlated noise for RF, as gaussian noise computation
        # is too time expensive because of computation of inverse and
        # determinant each time _corr is perturbed

        for i, target in enumerate(self.targets.targets):
            target_corrfix = corrfix[i]
            target_noise_corr = noise_corr[i]

            if not target_corrfix:
                # exponential for each target
                target.get_covariance = target.valuation.get_covariance_exp
                continue

            if (target_noise_corr == 0 and np.any(np.isnan(target.obsdata.yerr))):
                # diagonal for each target, corr inrelevant for likelihood, rel error
                target.get_covariance = target.valuation.get_covariance_nocorr
                continue

            elif target_noise_corr == 0:
                # diagonal for each target, corr inrelevant for likelihood
                target.get_covariance = target.valuation.get_covariance_nocorr_scalederr
                continue

            # gauss for RF
            if target.noiseref == 'rf':
                size = target.obsdata.x.size
                target.valuation.init_covariance_gauss(
                    target_noise_corr, size, rcond=rcond)
                target.get_covariance = target.valuation.get_covariance_gauss

            # exp for noise_corr
            elif target.noiseref == 'swd':
                target.get_covariance = target.valuation.get_covariance_exp

            else:
                message = 'The noise correlation automatically defaults to the \
exponential law. Explicitly state a noise reference for your user target \
(target.noiseref) if wished differently.'
                logger.info(message)
                target.noiseref == 'swd'
                target.get_covariance = target.valuation.get_covariance_exp

    def _init_chainarrays(self,sharedcurrentlikes, sharediiter):
        """from shared arrays"""
        ntargets = self.targets.ntargets
        chainidx = self.chainidx
        nchains = self.nchains

        accepted_models = int(self.iterations * np.max(self.acceptance) / 100.)
        self.nmodels = accepted_models *2 # 'iterations'

        msize = self.nmodels * self.maxlayers * 3
        nsize = self.nmodels * ntargets * 2
        missize = self.nmodels * (ntargets + 1)
        dtype = np.float32
            

	"""
        create new arrays to save PT values at temperature = 1,
        not share with baywatch
        """        

        currentlikes = np.frombuffer(sharedcurrentlikes, dtype=dtype)

	currentiiter = np.frombuffer(sharediiter, dtype=dtype)
        
	self.chaincurrentlikes = currentlikes#[chainidx]

        self.chaincurrentiiter = currentiiter
      

# update current model (change layer number and values)

    def _model_layerbirth(self, model):
        """
        Draw a random voronoi nucleus depth from z and assign a new Vs.

        The new Vs is based on the before Vs value at the drawn z_vnoi
        position (self.propdist[2]).
        """
        n, vs_vnoi, ra_vnoi, z_vnoi = Model.split_modelparams(model)

        # new voronoi depth
        zmin, zmax = self.priors['z']
        z_birth = self.rstate.uniform(low=zmin, high=zmax)

        ind = np.argmin((abs(z_vnoi - z_birth)))  # closest z_vnoi
        vs_before = vs_vnoi[ind]
        vs_birth = vs_before + self.rstate.normal(0, self.propdist[2])
       	
        ra_before = ra_vnoi[ind]
        if self.modifications_ra:
            ra_birth = ra_before + self.rstate.normal(0, self.propdist[6])
        else:
            ra_birth = ra_before

        z_new = np.concatenate((z_vnoi, [z_birth]))
        vs_new = np.concatenate((vs_vnoi, [vs_birth]))
        ra_new = np.concatenate((ra_vnoi, [ra_birth]))

        self.dvs2 = np.square(vs_birth - vs_before)
        self.dra2 = np.square(ra_birth - ra_before)
        return np.concatenate((vs_new, ra_new, z_new))

    def _model_layerdeath(self, model):
        """
        Remove a random voronoi nucleus depth from model. Delete corresponding
        Vs from model.
        """
        n, vs_vnoi, ra_vnoi, z_vnoi = Model.split_modelparams(model)
        ind_death = self.rstate.randint(low=0, high=(z_vnoi.size))
        z_before = z_vnoi[ind_death]
        vs_before = vs_vnoi[ind_death]
        ra_before = ra_vnoi[ind_death]

        z_new = np.delete(z_vnoi, ind_death)
        vs_new = np.delete(vs_vnoi, ind_death)
        ra_new = np.delete(ra_vnoi, ind_death)

        ind = np.argmin((abs(z_new - z_before)))
        vs_after = vs_new[ind]
        ra_after = ra_new[ind]
        self.dvs2 = np.square(vs_after - vs_before)
        self.dra2 = np.square(ra_after - ra_before)
        return np.concatenate((vs_new, ra_new, z_new))

    def _model_vschange(self, model):
        """Randomly chose a layer to change Vs with Gauss distribution."""
        ind = self.rstate.randint(0, model.size / 3)
        model1 = copy.deepcopy(model)
        vs_mod = self.rstate.normal(0, self.propdist[0])
        model[ind] = model[ind] + vs_mod
        vs_mod1 = self.rstate.normal(0, self.propdist[0]*0.9)
        model1[ind] = model1[ind] + vs_mod1
        self.m0m1_dvs2 = np.square(vs_mod)
        self.m1m2_dvs2 = np.square(vs_mod-vs_mod1) 
        return model, model1
        
    def _model_rachange(self, model):
        """Randomly chose a layer to change Vs with Gauss distribution."""
        ind = self.rstate.randint(model.size / 3, 2 * model.size / 3)
        model1 = copy.deepcopy(model)
	ra_mod = self.rstate.normal(0, self.propdist[5])
        model[ind] = model[ind] + ra_mod
	ra_mod1 = self.rstate.normal(0, self.propdist[5]*0.9)
        model1[ind] = model1[ind] + ra_mod1 
        self.m0m1_dra2 = np.square(ra_mod)
        self.m1m2_dra2 = np.square(ra_mod1-ra_mod)    
        return model, model1
        
    def _model_zvnoi_move(self, model):
        """Randomly chose a layer to change z_vnoi with Gauss distribution."""
        ind = self.rstate.randint(2 * model.size / 3, model.size)
        model1 = copy.deepcopy(model)
        z_mod = self.rstate.normal(0, self.propdist[1])
        model[ind] = model[ind] + z_mod
        z_mod1 = self.rstate.normal(0, self.propdist[1]*0.9)
        model1[ind] = model1[ind] + z_mod1
        self.m0m1_dzv2 = np.square(z_mod)
        self.m1m2_dzv2 = np.square(z_mod1-z_mod)                    
        return model, model1

    def _get_modelproposal(self, modify):
        model = copy.copy(self.currentmodel)

        if modify == 'vsmod':
            propmodel, drpropmodel = self._model_vschange(model)
        elif modify == 'ramod':
            propmodel, drpropmodel = self._model_rachange(model)
        elif modify == 'zvmod':
            propmodel, drpropmodel = self._model_zvnoi_move(model)
        elif modify == 'birth':
            propmodel = self._model_layerbirth(model)
        elif modify == 'death':
            propmodel = self._model_layerdeath(model)
            
	if modify in ['birth', 'death']:
        	return self._sort_modelproposal(propmodel), None
        else:
        	return self._sort_modelproposal(propmodel), self._sort_modelproposal(drpropmodel)

    def _sort_modelproposal(self, model):
        """
        Return the sorted proposal model.

        This method is necessary, if the z_vnoi from the new proposal model
        are not ordered, i.e. if one z_vnoi value is added or strongly modified.
        """
        n, vs, ra, z_vnoi = Model.split_modelparams(model)
        if np.all(np.diff(z_vnoi) > 0):   # monotone increasing
            return model
        else:
            ind = np.argsort(z_vnoi)
            model_sort = np.concatenate((vs[ind], ra[ind], z_vnoi[ind]))
        return model_sort

    def _validmodel(self, model):
        """
        Check model before the forward modeling.

        - The model must contain all values > 0.
        - The layer thicknesses must be at least thickmin km.
        - if lvz: low velocity zones are allowed with the deeper layer velocity
           no smaller than (1-perc) * velocity of layer above.
        - ... and some other constraints. E.g. vs boundaries (prior) given.
        """
        vp, vs, ra, h = Model.get_vp_vs_h(model, self.currentvpvs, self.mantle)

        # check whether nlayers lies within the prior
        layermin = self.priors['layers'][0]
        layermax = self.priors['layers'][1]
        layermodel = (h.size - 1)
        if not (layermodel >= layermin and layermodel <= layermax):
            logger.debug("chain%d: model- nlayers not in prior"
                         % self.chainidx)
            return False

        # check model for layers with thicknesses of smaller thickmin
        if np.any(h[:-1] < self.thickmin):
            logger.debug("chain%d: thicknesses are not larger than thickmin"
                         % self.chainidx)
            return False

        # check whether vs lies within the prior
        vsmin = self.priors['vs'][0]
        vsmax = self.priors['vs'][1]
        if np.any(vs < vsmin) or np.any(vs > vsmax):
            logger.debug("chain%d: model- vs not in prior"
                         % self.chainidx)
            return False
            
	# check whether ra lies within the prior
        if self.modifications_ra: # ra in modifications
            ramin = self.priors['ra'][0]
            ramax = self.priors['ra'][1]
            if np.any(ra < ramin) or np.any(ra > ramax):
                logger.debug("chain%d: model- ra not in prior"
                            % self.chainidx)
                return False     
                
        # check whether interfaces lie within prior
        zmin = self.priors['z'][0]
        zmax = self.priors['z'][1]
        z = np.cumsum(h)
        if np.any(z < zmin) or np.any(z > zmax):
            logger.debug("chain%d: model- z not in prior"
                         % self.chainidx)
            return False

        if self.lowvelperc is not None:
            # check model for low velocity zones. If larger than perc, then
            # compvels must be positive
            compvels = vs[1:] - (vs[:-1] * (1 - self.lowvelperc))
            if not compvels.size == compvels[compvels > 0].size:
                logger.debug("chain%d: low velocity zone issues"
                             % self.chainidx)
                return False

        if self.highvelperc is not None:
            # check model for high velocity zones. If larger than perc, then
            # compvels must be positive.
            compvels = (vs[:-1] * (1 + self.highvelperc)) - vs[1:]
            if not compvels.size == compvels[compvels > 0].size:
                logger.debug("chain%d: high velocity zone issues"
                             % self.chainidx)
                return False

        return True

    def _get_hyperparameter_proposal(self):
        noise = copy.copy(self.currentnoise)
        ind = self.rstate.choice(self.noiseinds)
       	noise_mod = self.rstate.normal(0, self.propdist[3])
        noise[ind] = noise[ind] + noise_mod
        return noise

    def _validnoise(self, noise):
        for idx in self.noiseinds:
            if noise[idx] < self.noisepriors[idx][0] or \
                    noise[idx] > self.noisepriors[idx][1]:
                return False
        return True

    def _get_vpvs_proposal(self):
        vpvs = copy.copy(self.currentvpvs)
        vpvs_mod = self.rstate.normal(0, self.propdist[4])
        vpvs = vpvs + vpvs_mod
        return vpvs

    def _validvpvs(self, vpvs):
        # only works if vpvs-priors is a range
        if vpvs < self.priors['vpvs'][0] or \
                vpvs > self.priors['vpvs'][1]:
            return False
        return True

    def _get_rj_proposal(self, modify, m2=False):
    	#delay rejection
    	if modify == 'vsmod':
    		propdist = self.propdist[0]
    		if m2:
    			delta = self.m1m2_dvs2
    		else: 
    			delta = self.m0m1_dvs2
    	elif modify == 'ramod':
    		propdist = self.propdist[5]
		if m2:
    			delta = self.m1m2_dra2
    		else: 
    			delta = self.m0m1_dra2
    	elif modify == 'zvmod':
    		propdist = self.propdist[1]
    		if m2:
    			delta = self.m1m2_dzv2
    		else: 
    			delta = self.m0m1_dzv2
    				
	B = delta / (2. * np.square(propdist))
	
        return  B *(-1.)
        
# accept / save current models

    def adjust_propdist(self):
        """
        Modify self.propdist to adjust acceptance rate of models to given
        percentace span: increase or decrease by five percent.
        """
        with np.errstate(invalid='ignore'):
            acceptrate = self.accepted / self.proposed * 100

        # minimum distribution width forced to be not less than 1 m/s, 1 m
        # actually only touched by vs distribution
        propdistmin = np.full(acceptrate.size, 0.001)

        for i, rate in enumerate(acceptrate):
            if np.isnan(rate):
                # only if not inverted for
                continue
            if rate < self.acceptance[0]:
                new = self.propdist[i] * 0.95
                if new < propdistmin[i]:
                    new = propdistmin[i]
                self.propdist[i] = new

            elif rate > self.acceptance[1]:
                self.propdist[i] = self.propdist[i] * 1.05
            else:
                pass

    def get_acceptance_probability(self, modify, likelihood_m1=None):
        """
        Acceptance probability will be computed dependent on the modification.

        Parametrization alteration (Vs or voronoi nuclei position)
            the acceptance probability is equal to likelihood ratio.

        Model dimension alteration (layer birth or death)
            the probability was computed after the formulation of Bodin et al.,
            2012: 'Transdimensional inversion of receiver functions and
            surface wave dispersion'.
        """
        T = (1/self.currenttmps)
        
        ###only no-dimensional moves are considered for delayed rejection
        if likelihood_m1 is not None:
            A = T*(self.targets.proposallikelihood - self.currentlikelihood)
            B = self._get_rj_proposal(modify, m2=True)
            C = self._get_rj_proposal(modify)
            diff_m1m2 = likelihood_m1-self.targets.proposallikelihood
            diff_m1m0 = likelihood_m1-self.currentlikelihood
            
            if  diff_m1m2 >= 710:
            	return '-inf'          
	    else:
	    	alpha1 = min(1, np.exp(diff_m1m2))
	    	if alpha1 == 1:
	    		return '-inf'       	
            
            #alpha 2 should > 2	
            if diff_m1m0 >= 710:
	    	alpha2 = 1 
	    	return '+inf'   
	    else:
	    	alpha2 = min(1, np.exp(diff_m1m0))
	    	if alpha2 ==1:
	    		return '+inf'
	    	
	    	
            D = (1 - alpha1**T)
            E = (1 - alpha2**T)
            alpha = A + B -C + np.log(D) - np.log(E)
            return alpha

        if modify in ['vsmod', 'ramod', 'zvmod', 'noise', 'vpvs']:
            # only velocity or thickness changes are made
            # also used for noise changes
            A= self.targets.proposallikelihood - self.currentlikelihood
            alpha = T*A
		
        elif modify in ['birth', ]:
            theta_vs = self.propdist[2]  # Gaussian distribution
            # self.dvs2 = delta vs square = np.square(v'_(k+1) - v_(i))
            A = (theta_vs * np.sqrt(2 * np.pi)) / self.dv
            B = self.dvs2 / (2. * np.square(theta_vs))
            C = self.targets.proposallikelihood - self.currentlikelihood
            
            if  self.modifications_ra:
            	theta_ra = self.propdist[6]  # Gaussian distribution
            	D = (theta_ra * np.sqrt(2 * np.pi)) / self.dra
            	E = self.dra2 / (2. * np.square(theta_ra))
            	alpha = np.log(A) + B + T*C + np.log(D) + E
            else:         	
           	alpha = np.log(A) + B +  T*C

        elif modify in ['death', ]:
            theta_vs = self.propdist[2]  # Gaussian distribution
            # self.dvs2 = delta vs square = np.square(v'_(j) - v_(i))
            A = self.dv / (theta_vs * np.sqrt(2 * np.pi))
            B = self.dvs2 / (2. * np.square(theta_vs))
            C = self.targets.proposallikelihood - self.currentlikelihood

            if  self.modifications_ra:
           	theta_ra = self.propdist[6]  # Gaussian distribution
           	D = self.dra / (theta_ra * np.sqrt(2 * np.pi))
           	E = self.dra2 / (2. * np.square(theta_ra))
           	alpha = np.log(A) - B + T*C+ np.log(D) - E
            else:          	
           	alpha = np.log(A) - B + T*C

        return alpha
        
    def accept_as_currentmodel(self, model, noise, vpvs):
        """Assign currentmodel and currentvalues to self."""
        self.currentmisfits = self.targets.proposalmisfits
        self.currentlikelihood = self.targets.proposallikelihood
        self.currentmodel = model
        self.currentnoise = noise
        self.currentvpvs = vpvs
        self.lastmoditer = self.iiter
        
    def append_currentmodel(self):
        """Append currentmodel to chainmodels and values."""

        self.n += 1
        
    def append_currenttmpmodel(self):
        """Append currentmodel to current chainmodels and values for usage of tmp."""
        self.partmp.append_tmpmodels(self.tmpiiter, self.chainidx, self.currentmodel,  		
        			self.currentmisfits, self.currentlikelihood, 
        			self.currentnoise, self.currentvpvs)
        
            
    def swap_tmp(self):            		
		 	
        if self.tmpiiter >= self.init_swap  and self.tmpiiter % self.freq_swap == 0 : #and \
            #self.tmpiiter <= self.iterations:
            chaintmp = np.nan
            
            while np.isnan(chaintmp):
            	chaintmp = self.partmp.get_swaptmp(self.tmpiiter, self.chainidx)
            	time.sleep(.5)
            self.currenttmps = chaintmp
 
    def convtest(self):
    	conv = np.nan
        while  np.isnan(conv) or ( conv != 0 and conv== self.nconv  ):
            conv = self.partmp.get_conv(self.chainidx)

	self.nconv = conv
	
        if conv >0 and conv <=2:
        	self.iiter = self.iiter -self.addiiter
        	self.iter_phase1 = self.iter_phase1 +  self.addiiter  
     	elif conv == 3:
     		logger.debug('chains are not covereged:increase iterations')
     		
                
    def invalid_model(self, modify):
        self.chaincurrentlikes[self.chainidx] = np.nan
        self.swap_tmp()
        logger.debug('Not able to find a proposal for %s' % modify)
        self.iiter +=  1

    def get_proposal(self):
        if self.iiter < (-self.iter_phase1 + (self.iterations * 0.01)):
            # only allow vs and z modifications the first 1 % of iterations
            if self.modifications_ra:
            	modify = self.rstate.choice(['vsmod', 'zvmod','ramod'] + self.noisemods +
                                        self.vpvsmods)
	    else:
                modify = self.rstate.choice(['vsmod', 'zvmod'] + self.noisemods +
                                            self.vpvsmods)
        else:
            modify = self.rstate.choice(self.modifications)

        if modify in self.modelmods:
            proposalmodel, drproposalmodel = self._get_modelproposal(modify)
            proposalnoise = self.currentnoise
            proposalvpvs = self.currentvpvs
            if not self._validmodel(proposalmodel):
                proposalmodel = None
            if modify in ['vsmod', 'zvmod','ramod'] and not self._validmodel(drproposalmodel):
            	#delay rejection only no dimension change move will be considered
                drproposalmodel = None

        elif modify in self.noisemods:
            proposalmodel = self.currentmodel
            drproposalmodel = self.currentmodel
            proposalnoise = self._get_hyperparameter_proposal()
            proposalvpvs = self.currentvpvs
            if not self._validnoise(proposalnoise):
                proposalmodel = None

        elif modify == 'vpvs':
            proposalmodel = self.currentmodel
            drproposalmodel = self.currentmodel
            proposalnoise = self.currentnoise
            proposalvpvs = self._get_vpvs_proposal()
            if not self._validvpvs(proposalvpvs):
                proposalmodel = None

        if proposalmodel is None:
            # If not a valid proposal model and noise params are found,
            # leave self.iterate and try with another modification
            # should not occur often.
            return modify, proposalmodel, proposalnoise, proposalvpvs, drproposalmodel
            
        # compute synthetic data and likelihood, misfit
        vp, vs, ra, h = Model.get_vp_vs_h(proposalmodel, proposalvpvs, self.mantle)
        vs_methods = self.priors['vs_methods']
        self.targets.evaluate(h=h, vp=vp, vs=vs, ra=ra, methods=vs_methods, noise=proposalnoise)
        return modify, proposalmodel, proposalnoise, proposalvpvs, drproposalmodel
        
# run optimization

    def iterate(self):
	self.chaincurrentiiter[self.chainidx] = self.tmpiiter
	
	modify, proposalmodel, proposalnoise, proposalvpvs, drproposalmodel = self.get_proposal()
    	if proposalmodel is None:  
            self.invalid_model(modify)
            return
            
        paridx = PAR_MAP[modify]
          
        # Replace self.currentmodel with proposalmodel with acceptance
        # probability alpha. Accept candidate sample (proposalmodel)
        # with probability alpha, or reject it with probability (1 - alpha).
        # these are log values ! alpha is log.
        u = np.log(self.rstate.uniform(0, 1))
        alpha = self.get_acceptance_probability(modify)
	
        if u < alpha:
            # always the case if self.jointlike > self.bestlike (alpha>1)
            self.accept_as_currentmodel(proposalmodel, proposalnoise, proposalvpvs)
            self.append_currentmodel()
            
            self.accepted[paridx] += 1
        elif modify in ['vsmod', 'zvmod','ramod']:
            ##delayed rejection, only condiser no-dimension change move
            ## save 1st model's likelihood for the calculation of alpha2
            likelihood_m1 = self.targets.proposallikelihood
            
            if drproposalmodel is None:  	
            	self.invalid_model(modify)
            	return
            			
            # compute synthetic data and likelihood, misfit
            vp, vs, ra, h = Model.get_vp_vs_h(drproposalmodel, proposalvpvs, self.mantle)
            vs_methods = self.priors['vs_methods']
            self.targets.evaluate(h=h, vp=vp, vs=vs, ra=ra, methods=vs_methods, noise=proposalnoise)
           
            u = np.log(self.rstate.uniform(0, 1))
            alpha2 = self.get_acceptance_probability(modify, likelihood_m1=likelihood_m1)
          
            if alpha2 != '-inf' and u < alpha2 or alpha2 == '+inf':
            	self.accept_as_currentmodel(drproposalmodel, proposalnoise, proposalvpvs)
            	self.append_currentmodel()
            	self.accepted[paridx] += 1
        self.proposed[paridx] += 1

	#swap temperature        
        self.chaincurrentlikes[self.chainidx] = self.currentlikelihood	       	
      	self.swap_tmp()
      	  	       	      		
        # print inversion status information
        if self.iiter % 5000 == 0:
            runtime = time.time() - self.tnull
            current_iterations = self.iiter + self.iter_phase1
                                
            if current_iterations > 0:
                acceptrate = float(self.n) / current_iterations * 100.

                logger.info('%6d %5d + hs %8.3f\t%9d |%6.1f s  | %.1f ' % (
                    self.lastmoditer, self.currentmodel.size/2 - 1,
                    self.currentmisfits[-1], self.currentlikelihood,
                    runtime, acceptrate) + r'%')

            self.tnull = time.time()

        # stabilize model acceptance rate
        if self.iiter % 1000 == 0:
            if np.all(self.proposed) != 0:
                self.adjust_propdist()

        self.iiter += 1
        

    def run_chain(self):
        t0 = time.time()
        self.tnull = time.time()
        self.iiter = -self.iter_phase1
        self.currenttmps = self.partmp._get_init_tmp(self.chainidx)
	self.tmpiiter = 0# tmp models counter
	self.append_currenttmpmodel()
	self.tmpiiter = 1
	self.nconv = 0
        
        self.modelmods = ['vsmod', 'zvmod', 'birth', 'death', 'ramod'] if self.modifications_ra \
            else ['vsmod', 'zvmod', 'birth', 'death']
        self.noisemods = [] if len(self.noiseinds) == 0 else ['noise']
        self.vpvsmods = [] if type(self.priors['vpvs']) == np.float else ['vpvs']
        self.modifications = self.modelmods + self.noisemods + self.vpvsmods

        self.accepted = np.zeros(len(self.propdist))
        self.proposed = np.zeros(len(self.propdist))

        while self.iiter < self.iter_phase2-1:
            self.iterate()
            self.append_currenttmpmodel()
            self.tmpiiter += 1
            # convergence test
      	    #if self.tmpiiter == self.iter_phase1:
      		#self.convtest()
        

        runtime = (time.time() - t0)  
        logger.debug('time for inversion: %.2f s' % runtime)


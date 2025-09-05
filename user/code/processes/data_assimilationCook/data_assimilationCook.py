#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

# from .utils import compute_rms_std_optimization
# from .optimize.initialize import optimize_initialize
# from .optimize.update import optimize_update
# from .optimize.update_lbfgs import optimize_update_lbfgs
# from .outputs.output_ncdf import update_ncdf_optimize, output_ncdf_optimize_final
# from .outputs.prints import print_costs, save_rms_std, print_info_data_assimilation
# from .outputs.plots import update_plot_inversion, plot_cost_functions

from igm.processes.iceflow.emulate.emulate import update_iceflow_emulator, update_iceflow_emulated
#from igm.processes.iceflow.emulate.emulator import update_iceflow_emulator
#from igm.processes.iceflow.emulate.emulated import update_iceflow_emulated
from igm.processes.iceflow import initialize as iceflow_initialize
from igm.utils.gradient.compute_divflux import compute_divflux
from igm.utils.math.getmag import getmag
from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from scipy import stats
from netCDF4 import Dataset
import os, copy
import matplotlib
import datetime
from tqdm import tqdm

def initialize(cfg, state):

    iceflow_initialize(cfg, state) # initialize the iceflow model

    optimize_initialize(cfg, state)

    # update_iceflow_emulator(cfg, state, 0) # initialize the emulator
  
    # iterate over the optimization process
    for i in range(cfg.processes.data_assimilationCook.optimization.nbitmax+1):

        cost = {}

        if cfg.processes.data_assimilationCook.optimization.method == "ADAM":
            optimize_update(cfg, state, cost, i)
        elif cfg.processes.data_assimilationCook.optimization.method == "L-BFGS":
            optimize_update_lbfgs(cfg, state, cost, i)
        else:
            raise ValueError(f"Unknown optim. method: {cfg.processes.data_assimilationCook.optimization.method}")

        compute_rms_std_optimization(state, i)
            
        # retraning the iceflow emulator
        if cfg.processes.data_assimilationCook.optimization.retrain_iceflow_model:
            update_iceflow_emulator(cfg, state, i+1, 
                                    pertubate=cfg.processes.data_assimilationCook.optimization.pertubate) 
            cost["glen"] = state.COST_EMULATOR[-1]
            
        print_costs(cfg, state, cost, i)
        print_info_data_assimilation(cfg, state,  cost, i)

        if i % cfg.processes.data_assimilationCook.output.freq == 0:
            if cfg.processes.data_assimilationCook.output.plot2d:
                update_plot_inversion(cfg, state, i)
            if cfg.processes.data_assimilationCook.output.save_iterat_in_ncdf:
                update_ncdf_optimize(cfg, state, i)

            # stopping criterion: stop if the cost no longer decrease
            # if i>cfg.processes.data_assimilationCook.optimization.nbitmin:
            #     cost = [c[0] for c in costs]
            #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
            #         break;  

    # now that the ice thickness is optimized, we can fix the bed once for all! (ONLY FOR GROUNDED ICE)
    state.topg = state.usurf - state.thk

    if not cfg.processes.data_assimilationCook.output.save_result_in_ncdf=="":
        output_ncdf_optimize_final(cfg, state)

    plot_cost_functions() # ! Bug right now with plotting values... (extra headers)

    save_rms_std(cfg, state) 

def update(cfg, state):
    pass

def finalize(cfg, state):
    pass

def compute_rms_std_optimization(state, i):
    I = state.icemaskobs > 0.5

    if i == 0:
        state.rmsthk = []
        state.stdthk = []
        state.rmsvel = []
        state.stdvel = []
        state.rmsusurf = []
        state.stdusurf = []
        state.rmsdiv = []
        state.stddiv = []

    if hasattr(state, "thkobs"):
        ACT = ~tf.math.is_nan(state.thkobs)
        if np.sum(ACT) == 0:
            state.rmsthk.append(0)
            state.stdthk.append(0)
        else:
            state.rmsthk.append(np.nanmean(state.thk[ACT] - state.thkobs[ACT]))
            state.stdthk.append(np.nanstd(state.thk[ACT] - state.thkobs[ACT]))

    else:
        state.rmsthk.append(0)
        state.stdthk.append(0)

    if hasattr(state, "uvelsurfobs"):
        velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
        ACT = ~np.isnan(velsurfobs_mag)

        state.rmsvel.append(
            np.mean(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
        state.stdvel.append(
            np.std(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
    else:
        state.rmsvel.append(0)
        state.stdvel.append(0)

    if hasattr(state, "divfluxobs"):
        state.rmsdiv.append(np.mean(state.divfluxobs[I] - state.divflux[I]))
        state.stddiv.append(np.std(state.divfluxobs[I] - state.divflux[I]))
    else:
        state.rmsdiv.append(0)
        state.stddiv.append(0)

    if hasattr(state, "usurfobs"):
        state.rmsusurf.append(np.mean(state.usurf[I] - state.usurfobs[I]))
        state.stdusurf.append(np.std(state.usurf[I] - state.usurfobs[I]))
    else:
        state.rmsusurf.append(0)
        state.stdusurf.append(0)
 
def create_density_matrix(data, kernel_size):
    # Convert data to binary mask (1 for valid data, 0 for NaN)
    binary_mask = tf.where(tf.math.is_nan(data), tf.zeros_like(data), tf.ones_like(data))

    # Create a kernel for convolution (all ones)
    kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=binary_mask.dtype)

    # Apply convolution to count valid data points in the neighborhood
    density = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(binary_mask, 0), -1), 
                           kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Remove the extra dimensions added for convolution
    density = tf.squeeze(density)

    return density
 
def compute_flow_direction_for_anisotropic_smoothing(state):
    uvelsurf = tf.where(tf.math.is_nan(state.uvelsurf), 0.0, state.uvelsurf)
    vvelsurf = tf.where(tf.math.is_nan(state.vvelsurf), 0.0, state.vvelsurf)

    state.flowdirx = (
        uvelsurf[1:, 1:] + uvelsurf[:-1, 1:] + uvelsurf[1:, :-1] + uvelsurf[:-1, :-1]
    ) / 4.0
    state.flowdiry = (
        vvelsurf[1:, 1:] + vvelsurf[:-1, 1:] + vvelsurf[1:, :-1] + vvelsurf[:-1, :-1]
    ) / 4.0

    from scipy.ndimage import gaussian_filter

    state.flowdirx = gaussian_filter(state.flowdirx, 3, mode="constant")
    state.flowdiry = gaussian_filter(state.flowdiry, 3, mode="constant")

    # Same as gaussian filter above but for tensorflow is (NOT TESTED)
    # import tensorflow_addons as tfa
    # state.flowdirx = ( tfa.image.gaussian_filter2d( state.flowdirx , sigma=3, filter_shape=100, padding="CONSTANT") )

    state.flowdirx /= getmag(state.flowdirx, state.flowdiry)
    state.flowdiry /= getmag(state.flowdirx, state.flowdiry)

    state.flowdirx = tf.where(tf.math.is_nan(state.flowdirx), 0.0, state.flowdirx)
    state.flowdiry = tf.where(tf.math.is_nan(state.flowdiry), 0.0, state.flowdiry)
    
    # state.flowdirx = tf.zeros_like(state.flowdirx)
    # state.flowdiry = tf.ones_like(state.flowdiry)

    # this is to plot the observed flow directions
    # fig, axs = plt.subplots(1, 1, figsize=(8,16))
    # plt.quiver(state.flowdirx,state.flowdiry)
    # axs.axis("equal")

def infer_params_cook(state, cfg):
    #This function allows for both parameters to be specified as varying 2D fields (you could compute them pixel-wise from VelMag by swapping in VelMag for VelPerc).
    #This is probably not a good idea, because the values of both parameters do not depend solely on the velocity at that point. But feel free to try! If you do
    #want to do that, you'll also need to un-comment the code block for smoothing and then converting the smoothed weights back to tensors (you may also want to
    #visualise the figures!), and set up a state.convexity_weights field to act as the 2D array
    import scipy
    
    #cfg.processes.data_assimilationCook.optimization.nbitmax = 1500
    RGIRegion = int(cfg.inputs.oggm_shopCook.RGI_ID[15:17])
    
    #Get list of G entities in each C/get multi-valued ice mask
    #Loop over all Gs to construct parameter rasters
    
    # params.opti_divfluxobs_std = 1.0
    # params.opti_usurfobs_std = 0.3
    # params.opti_regu_param_thk = 1.0
    
    percthreshold = 99
    NumGlaciers = int(tf.reduce_max(state.icemask).numpy())
    
    AreaCorrection = [-0.24,-0.53,-0.066,-0.080,-0.019,-0.3,-0.23,-0.29,-0.23,-0.48,-1.2,-0.58,-0.28,-0.39,0.40,-1.6,-0.18,-0.69,-0.11] #Li et al. (2019)
    
    TotalArea = tf.reduce_sum(tf.where(state.icemask > 0.5,1.0,0.0))*state.dx**2
    TotalArea = TotalArea/1e6
    TotalArea = TotalArea + (TotalArea*(AreaCorrection[RGIRegion-1]/100)*10)
    TotalVolume = 0.0
    VolBasins = 0.0
    
    state.volumes = tf.experimental.numpy.copy(state.thk)
    state.volume_weights = tf.experimental.numpy.copy(state.thk)
    state.volume_weights = tf.where(state.icemaskobs > 0.0, cfg.processes.data_assimilationCook.cook.vol_std, 0.0)
    
    #Get some initial information
    VelMag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    VelMag = tf.where(tf.math.is_nan(VelMag),1e-6,VelMag)
    VelMag = tf.where(VelMag==0,1e-6,VelMag)
    
    #Start of G loop
    for i in range(1,NumGlaciers+1):
        
        #Get area predictor
        Area = tf.reduce_sum(tf.where(state.icemask==i,1.0,0.0))*state.dx**2
        Area = np.round(Area.numpy()/1000**2, decimals=1)
        Area = Area + (Area*(AreaCorrection[RGIRegion-1]/100)*10)
        #print('Area is: ', Area)
        #print('Predicted volume is: ', np.exp(np.log(Area)*1.359622487))
        #Work out nominal volume target based on volume-area scaling - only has much of an effect if no other observations
        if (NumGlaciers/TotalArea < 0.1) & (NumGlaciers > 4) & (TotalArea > 100):
            state.volumes = tf.where(state.icemaskobs > 0.5, 0.034*TotalArea**1.25, state.volumes)
            print('Target volume is: '+str(0.034*TotalArea**1.25))
            #print(0.034*TotalArea**1.25)
            # if cfg.processes.data_assimilationCook.cook.vol_std == 0.0:
            #     state.volume_weights = tf.where(state.icemaskobs > 0.5, 0.0, state.volume_weights) #1.0
            # else:
            #     state.volume_weights = tf.where(state.icemaskobs > 0.5, 1.0, state.volume_weights) #1.0
            # cfg.processes.data_assimilationCook.fitting.divfluxobs_std = 10.0
            # cfg.processes.data_assimilationCook.fitting.usurfobs_std = 3.0
            # cfg.processes.data_assimilationCook.regularization.thk = 1.0 #1.0
            #params.opti_velsurfobs_std = 3.0
        else:
            VolBasins = VolBasins + 0.034*Area**1.375
            print('Target volume is: '+str(VolBasins))
            state.volumes = tf.where(state.icemaskobs > 0.5, VolBasins, state.volumes) #Make sure to put into km3!
        #print('Volume: ',Area,0.034*Area**1.375)
        if Area <= 0.000:
            continue
        
        #Get velocity predictors
        VelMean = np.round(np.mean(VelMag[state.icemaskobs==i]),decimals=2)
        #print("Mean velocity is: ", VelMean)
        
        if VelMean == 0.0:
            print('No velocity')
            #With volume-area scaling (on the assumption these will all be very small)
            # state.slidingco = tf.where(state.icemaskobs == i, 0.1, state.slidingco)
            # state.uvelsurfobs = tf.where(state.icemaskobs > 0.5, -10.0, state.uvelsurfobs)
            # state.vvelsurfobs = tf.where(state.icemaskobs > 0.5, 10.0, state.vvelsurfobs)
            # cfg.processes.data_assimilationCook.control_list = ['thk', 'usurf']
            # state.uvelsurfobs = tf.where(state.icemaskobs == i, np.nan, state.uvelsurfobs)
            # state.vvelsurfobs = tf.where(state.icemaskobs == i, np.nan, state.vvelsurfobs)
            #cfg.processes.data_assimilationCook.fitting.velsurfobs_std = 3.0
        
    VelMean = np.round(np.mean(VelMag[state.icemaskobs > 0.5]),decimals=2)
    if VelMean == 0.0:
        print('No velocity anywhere: '+cfg.inputs.oggm_shopCook.RGI_ID)
        cfg.processes.data_assimilationCook.optimization.nbitmax = 50
        state.volume_weights = tf.where(state.icemaskobs > 0.5, 0.00001, state.volume_weights)
            
def total_cost(cfg, state, cost, i):

    # misfit between surface velocity
    if "velsurf" in cfg.processes.data_assimilationCook.cost_list:
        cost["velsurf"] = misfit_velsurf(cfg,state)

    # misfit between ice thickness profiles
    if "thk" in cfg.processes.data_assimilationCook.cost_list:
        cost["thk"] = misfit_thk(cfg, state)

    # misfit between divergence of flux
    if ("divfluxfcz" in cfg.processes.data_assimilationCook.cost_list):
        cost["divflux"] = cost_divfluxfcz(cfg, state, i)
    elif ("divfluxobs" in cfg.processes.data_assimilationCook.cost_list):
        cost["divflux"] = cost_divfluxobs(cfg, state, i)

    # misfit between top ice surfaces
    if "usurf" in cfg.processes.data_assimilationCook.cost_list:
        cost["usurf"] = misfit_usurf(cfg, state) 

    # add penalty terms to force obstacle constraints
    if "penalty" in cfg.processes.data_assimilationCook.optimization.obstacle_constraint:

        # force zero thikness outisde the mask
        if "icemask" in cfg.processes.data_assimilationCook.cost_list:
            cost["icemask"] = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )

        # Here one enforces non-negative ice thickness
        if "thk" in cfg.processes.data_assimilationCook.control_list:
            cost["thk_positive"] = \
            10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )

        # Here one enforces non-negative slidinco
        if ("slidingco" in cfg.processes.data_assimilationCook.control_list) & \
            (not cfg.processes.data_assimilationCook.fitting.log_slidingco):
            cost["slidingco_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.slidingco >= 0, 0.0, state.slidingco**2) ) 

        # Here one enforces non-negative arrhenius
        if ("arrhenius" in cfg.processes.data_assimilationCook.control_list):
            cost["arrhenius_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 1, 0.0, state.arrhenius**2) ) 
        
    if cfg.processes.data_assimilationCook.cook.infer_params:
        cost["volume"] = cost_vol(cfg, state)

    # Here one adds a regularization terms for the bed toporgraphy to the cost function
    if "thk" in cfg.processes.data_assimilationCook.control_list:
        cost["thk_regu"] = regu_thk(cfg, state)

    # Here one adds a regularization terms for slidingco to the cost function
    if "slidingco" in cfg.processes.data_assimilationCook.control_list:
        cost["slid_regu"] = regu_slidingco(cfg, state)

    # Here one adds a regularization terms for arrhenius to the cost function
    if "arrhenius" in cfg.processes.data_assimilationCook.control_list:
        cost["arrh_regu"] = regu_arrhenius(cfg, state) 

    return tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))

def cost_divfluxfcz(cfg,state,i):

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, method=cfg.processes.data_assimilationCook.divflux.method
    )
 
    ACT = state.icemaskobs > 0.5
    if i % 10 == 0:
        # his does not need to be comptued any iteration as this is expensive
        state.res = stats.linregress(
            state.usurf[ACT], divflux[ACT]
        )  # this is a linear regression (usually that's enough)
    # or you may go for polynomial fit (more gl, but may leads to errors)
    #  weights = np.polyfit(state.usurf[ACT],divflux[ACT], 2)
    divfluxtar = tf.where(
        ACT, state.res.intercept + state.res.slope * state.usurf, 0.0
    )
#   divfluxtar = tf.where(ACT, np.poly1d(weights)(state.usurf) , 0.0 )
    
    ACT = state.icemaskobs > 0.5
    COST_D = 0.5 * tf.reduce_mean(
        ((divfluxtar[ACT] - divflux[ACT]) / cfg.processes.data_assimilationCook.fitting.divfluxobs_std) ** 2
    )

    if cfg.processes.data_assimilationCook.divflux.force_zero_sum:
            ACT = state.icemaskobs > 0.5
            COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / cfg.processes.data_assimilationCook.fitting.divfluxobs_std) ** 2
            
    if tf.math.is_nan(COST_D):
        COST_D = tf.Variable(0.0)
        divflux = tf.where(tf.math.is_nan(divflux), 0.0, divflux)

    return COST_D
 
def cost_divfluxobs(cfg,state,i):

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, method=cfg.processes.data_assimilationCook.divflux.method
    )
 
    divfluxtar = state.divfluxobs
    ACT = ~tf.math.is_nan(divfluxtar)
    COST_D = 0.5 * tf.reduce_mean(
        ((divfluxtar[ACT] - divflux[ACT]) / cfg.processes.data_assimilationCook.fitting.divfluxobs_std) ** 2
    )
 
    dddx = (divflux[:, 1:] - divflux[:, :-1])/state.dx
    dddy = (divflux[1:, :] - divflux[:-1, :])/state.dx
    COST_D += (cfg.processes.data_assimilationCook.regularization.divflux) * 0.5 * ( tf.reduce_mean(dddx**2) + tf.reduce_mean(dddy**2) )

    if cfg.processes.data_assimilationCook.divflux.force_zero_sum:
        ACT = state.icemaskobs > 0.5
        COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / cfg.processes.data_assimilationCook.fitting.divfluxobs_std) ** 2

    return COST_D

def cost_vol(cfg,state):

    ACT = state.icemaskobs > 0.5
    
    ModVols = tf.experimental.numpy.copy(state.icemaskobs)
    
    ModVols = tf.where(ModVols>0.5,(tf.reduce_sum(tf.where(state.icemask>0.5,state.thk,0.0))*state.dx**2)/1e9,ModVols)

    cost = 0.5 * tf.reduce_mean(
           ( (state.volumes[ACT] - ModVols[ACT]) / state.volume_weights[ACT]  )** 2
    )
    if tf.math.is_nan(cost) or tf.math.is_inf(cost):
        cost = tf.Variable(0.0)
    return cost

def misfit_thk(cfg,state):

    ACT = ~tf.math.is_nan(state.thkobs)

    return 0.5 * tf.reduce_mean( state.dens_thkobs[ACT] * 
        ((state.thkobs[ACT] - state.thk[ACT]) / cfg.processes.data_assimilationCook.fitting.thkobs_std) ** 2
    )

def misfit_usurf(cfg,state):

    ACT = state.icemaskobs > 0.5

    return 0.5 * tf.reduce_mean(
        (
            (state.usurf[ACT] - state.usurfobs[ACT])
            / cfg.processes.data_assimilationCook.fitting.usurfobs_std
        )
        ** 2
    )

def misfit_velsurf(cfg,state):    

    velsurf    = tf.stack([state.uvelsurf,    state.vvelsurf],    axis=-1) 
    velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

    REL = tf.expand_dims( (tf.norm(velsurfobs,axis=-1) >= cfg.processes.data_assimilationCook.fitting.velsurfobs_thr ) , axis=-1)

    ACT = ~tf.math.is_nan(velsurfobs) 

    cost = 0.5 * tf.reduce_mean(
           ( (velsurfobs[ACT & REL] - velsurf[ACT & REL]) / cfg.processes.data_assimilationCook.fitting.velsurfobs_std  )** 2
    )

    if tf.math.is_nan(cost):
       cost = tf.Variable(0.0)

    if cfg.processes.data_assimilationCook.fitting.include_low_speed_term:

        # This terms penalize the cost function when the velocity is low
        # Reference : Inversion of basal friction in Antarctica using exact and incompleteadjoints of a higher-order model
        # M. Morlighem, H. Seroussi, E. Larour, and E. Rignot, JGR, 2013
        cost += 0.5 * 100 * tf.reduce_mean(
            tf.math.log( (tf.norm(velsurf[ACT],axis=-1)+1) / (tf.norm(velsurfobs[ACT],axis=-1)+1) )** 2
        )

    return cost

def regu_arrhenius(cfg,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.arrhenius[:, 1:] - state.arrhenius[:, :-1])/state.dx
    dady = (state.arrhenius[1:, :] - state.arrhenius[:-1, :])/state.dx

    if cfg.processes.data_assimilationCook.optimization.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)
    
    if cfg.processes.data_assimilationCook.optimization.fix_opti_normalization_issue:
        REGU_S = (cfg.processes.data_assimilationCook.regularization.arrhenius) * 0.5 * (
            tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
        )
    else:
        REGU_S = (cfg.processes.data_assimilationCook.regularization.arrhenius) * (
            tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
        )

    return REGU_S

def regu_slidingco(cfg,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
    dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx

    if cfg.processes.data_assimilationCook.optimization.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)

    if cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor_sl == 1:
        if cfg.processes.data_assimilationCook.optimization.fix_opti_normalization_issue:
            REGU_S = (cfg.processes.data_assimilationCook.regularization.slidingco) * 0.5 * (
                tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
            )
        else:
            REGU_S = (cfg.processes.data_assimilationCook.regularization.slidingco) * (
                tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
            )
    else:
        dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
        dadx = (dadx[1:, :] + dadx[:-1, :]) / 2.0
        dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx
        dady = (dady[:, 1:] + dady[:, :-1]) / 2.0
 
        if cfg.processes.data_assimilationCook.optimization.sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            dadx = tf.where( MASK, dadx, 0.0)
            dady = tf.where( MASK, dady, 0.0)
 
        if cfg.processes.data_assimilationCook.optimization.fix_opti_normalization_issue:
            REGU_S = (cfg.processes.data_assimilationCook.regularization.slidingco) * 0.5 * (
                (1.0/np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor_sl))
                * tf.math.reduce_mean((dadx * state.flowdirx + dady * state.flowdiry)**2)
                + np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor_sl)
                * tf.math.reduce_mean((dadx * state.flowdiry - dady * state.flowdirx)**2)
            )
        else:
            REGU_S = (cfg.processes.data_assimilationCook.regularization.slidingco) * (
                (1.0/np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor_sl))
                * tf.nn.l2_loss((dadx * state.flowdirx + dady * state.flowdiry))
                + np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor_sl)
                * tf.nn.l2_loss((dadx * state.flowdiry - dady * state.flowdirx)) )

    return REGU_S

def regu_thk(cfg,state):

    areaicemask = tf.reduce_sum(tf.where(state.icemask>0.5,1.0,0.0))*state.dx**2

    # here we had factor 8*np.pi*0.04, which is equal to 1
    gamma = cfg.processes.data_assimilationCook.regularization.convexity_weight * areaicemask**(cfg.processes.data_assimilationCook.regularization.convexity_power-2.0)

    if cfg.processes.data_assimilationCook.regularization.to_regularize == 'topg':
        field = state.usurf - state.thk
    elif cfg.processes.data_assimilationCook.regularization.to_regularize == 'thk':
        field = state.thk

    if cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor == 1:
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdy = (field[1:, :] - field[:-1, :])/state.dx

        if cfg.processes.data_assimilationCook.optimization.sole_mask:
            dbdx = tf.where( (state.icemaskobs[:, 1:] > 0.5) & (state.icemaskobs[:, :-1] > 0.5) , dbdx, 0.0)
            dbdy = tf.where( (state.icemaskobs[1:, :] > 0.5) & (state.icemaskobs[:-1, :] > 0.5) , dbdy, 0.0)

        if cfg.processes.data_assimilationCook.optimization.fix_opti_normalization_issue:
            REGU_H = (cfg.processes.data_assimilationCook.regularization.thk) * 0.5 * (
                tf.math.reduce_mean(dbdx**2) + tf.math.reduce_mean(dbdy**2)
                - gamma * tf.math.reduce_mean(state.thk)
            )
        else:
            REGU_H = (cfg.processes.data_assimilationCook.regularization.thk) * (
                tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
                - gamma * tf.math.reduce_sum(state.thk)
            )
    else:
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
        dbdy = (field[1:, :] - field[:-1, :])/state.dx
        dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0

        if cfg.processes.data_assimilationCook.optimization.sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            dbdx = tf.where( MASK, dbdx, 0.0)
            dbdy = tf.where( MASK, dbdy, 0.0)
 
        if cfg.processes.data_assimilationCook.optimization.fix_opti_normalization_issue:
            REGU_H = (cfg.processes.data_assimilationCook.regularization.thk) * 0.5 * (
                (1.0/np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor))
                * tf.math.reduce_mean((dbdx * state.flowdirx + dbdy * state.flowdiry)**2)
                + np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor)
                * tf.math.reduce_mean((dbdx * state.flowdiry - dbdy * state.flowdirx)**2)
                - tf.math.reduce_mean(gamma*state.thk)
            )
        else:
            REGU_H = (cfg.processes.data_assimilationCook.regularization.thk) * (
                (1.0/np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor))
                * tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
                + np.sqrt(cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor)
                * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
                - tf.math.reduce_sum(gamma*state.thk)
            )

    return REGU_H

def optimize_initialize(cfg, state):

    ###### PERFORM CHECKS PRIOR OPTIMIZATIONS
    # if cfg.processes.data_assimilationCook.cook.infer_params:
    #     cfg.processes.data_assimilationCook.cost_list = ["velsurf", "usurf", "icemask", "thk"]

    # from scipy.ndimage import gaussian_filter
    # state.usurfobs = tf.Variable(gaussian_filter(state.usurfobs.numpy(), 3, mode="reflect"))
    # state.usurf    = tf.Variable(gaussian_filter(state.usurf.numpy(), 3, mode="reflect"))

    assert ("usurf" in cfg.processes.data_assimilationCook.cost_list) == ("usurf" in cfg.processes.data_assimilationCook.control_list)

    # make sure that there are least some profiles in thkobs
    if tf.reduce_all(tf.math.is_nan(state.thkobs)):
        if "thk" in cfg.processes.data_assimilationCook.cost_list:
            cfg.processes.data_assimilationCook.cost_list.remove("thk")

    ###### PREPARE DATA PRIOR OPTIMIZATIONS
 
    if "divfluxobs" in cfg.processes.data_assimilationCook.cost_list:
        if not hasattr(state, "divfluxobs"):
            state.divfluxobs = state.smb - state.dhdt

    if hasattr(state, "thkinit"):
        state.thkinit = tf.where(tf.math.is_nan(state.thkinit),0.0,state.thkinit)
        if tf.math.is_inf(tf.math.reduce_sum(state.thkinit)):
            state.thkinit = tf.where(state.icemask > 0.5, 10.0, state.thkinit)
        if tf.math.reduce_sum(state.thkinit)==0.0:
            state.thkinit = tf.where(state.icemask > 0.5, 10.0, state.thkinit)
        state.thk = state.thkinit
    else:
        state.thk = tf.zeros_like(state.thk)

    if cfg.processes.data_assimilationCook.optimization.init_zero_thk:
        state.thk = state.thk*0.0
        
    # this is a density matrix that will be used to weight the cost function
    if cfg.processes.data_assimilationCook.fitting.uniformize_thkobs:
        state.dens_thkobs = create_density_matrix(state.thkobs, kernel_size=5)
        state.dens_thkobs = tf.where(state.dens_thkobs>0, 1.0/state.dens_thkobs, 0.0)
        state.dens_thkobs = tf.where(tf.math.is_nan(state.thkobs),0.0,state.dens_thkobs)
        state.dens_thkobs = state.dens_thkobs / tf.reduce_mean(state.dens_thkobs[state.dens_thkobs>0])
    else:
        state.dens_thkobs = tf.ones_like(state.thkobs)
        
    # force zero slidingco in the floating areas
    #state.slidingco = tf.where( state.icemaskobs == 2, 0.0, state.slidingco)
    
    # this will infer values for slidingco and convexity weight based on the ice velocity and an empirical relationship from test glaciers with thickness profiles
    if cfg.processes.data_assimilationCook.cook.infer_params:
        #Because OGGM will index icemask from 0
        dummy = infer_params_cook(state, cfg)
        if tf.reduce_max(state.icemask).numpy() < 1:
            return
    
    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        state.optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.processes.data_assimilationCook.optimization.step_size,
            epsilon=cfg.processes.data_assimilationCook.optimization.optimizer_epsilon,
            clipnorm=cfg.processes.data_assimilationCook.optimization.optimizer_clipnorm
            )
    else:
        state.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=cfg.processes.data_assimilationCook.optimization.step_size,
            epsilon=cfg.processes.data_assimilationCook.optimization.optimizer_epsilon,
            clipnorm=cfg.processes.data_assimilationCook.optimization.optimizer_clipnorm
        )
        
def optimize_update(cfg, state, cost, i):

    sc = {}
    sc["thk"] = cfg.processes.data_assimilationCook.scaling.thk
    sc["usurf"] = cfg.processes.data_assimilationCook.scaling.usurf
    sc["slidingco"] = cfg.processes.data_assimilationCook.scaling.slidingco
    sc["arrhenius"] = cfg.processes.data_assimilationCook.scaling.arrhenius

    for f in cfg.processes.data_assimilationCook.control_list:
        if cfg.processes.data_assimilationCook.fitting.log_slidingco & (f == "slidingco"):
            vars(state)[f+'_sc'] = tf.Variable( tf.sqrt(vars(state)[f] / sc[f]) ) 
        else:
            vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f]) 

    with tf.GradientTape() as t:

        if cfg.processes.data_assimilationCook.optimization.step_size_decay < 1:
            state.optimizer.lr = cfg.processes.data_assimilationCook.optimization.step_size * (cfg.processes.data_assimilationCook.optimization.step_size_decay ** (i / 100))

        # is necessary to remember all operation to derive the gradients w.r.t. control variables
        for f in cfg.processes.data_assimilationCook.control_list:
            t.watch(vars(state)[f+'_sc'])

        for f in cfg.processes.data_assimilationCook.control_list:
            if cfg.processes.data_assimilationCook.fitting.log_slidingco & (f == "slidingco"):
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        update_iceflow_emulated(cfg, state)

        if not cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor == 1:
            compute_flow_direction_for_anisotropic_smoothing(state)
                
        cost_total = total_cost(cfg, state, cost, i)

        var_to_opti = [ ]
        for f in cfg.processes.data_assimilationCook.control_list:
            var_to_opti.append(vars(state)[f+'_sc'])

        # Compute gradient of COST w.r.t. X
        grads = tf.Variable(t.gradient(cost_total, var_to_opti))

        # this serve to restict the optimization of controls to the mask
        if cfg.processes.data_assimilationCook.optimization.sole_mask:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.data_assimilationCook.control_list[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                else:
                    grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
        else:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.data_assimilationCook.control_list[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

        # One step of descent -> this will update input variable X
        state.optimizer.apply_gradients(
            zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
        )

        ###################

        # get back optimized variables in the pool of state.variables
        for f in cfg.processes.data_assimilationCook.control_list:
            if cfg.processes.data_assimilationCook.fitting.log_slidingco & (f == "slidingco"):
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        # add reprojection step to force obstacle constraints
        if "reproject" in cfg.processes.data_assimilationCook.optimization.obstacle_constraint:

            if "icemask" in cfg.processes.data_assimilationCook.cost_list:
                state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)

            if "thk" in cfg.processes.data_assimilationCook.control_list:
                state.thk = tf.where(state.thk < 0, 0, state.thk)

            if "slidingco" in cfg.processes.data_assimilationCook.control_list:
                state.slidingco = tf.where(state.slidingco < 0, 0, state.slidingco)

            if "arrhenius" in cfg.processes.data_assimilationCook.control_list:
                # Here we assume a minimum value of 1.0 for the arrhenius factor (should not be hard-coded)
                state.arrhenius = tf.where(state.arrhenius < 1.0, 1.0, state.arrhenius) 

        state.divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, 
            method=cfg.processes.data_assimilationCook.divflux.method
        )

        # relaxation = 0.02
        # if relaxation>0:
        #     state.thk = tf.maximum(state.thk + relaxation * gaussian_filter_tf(state.divflux, sigma=2.0, kernel_size=13), 0)
        #     state.usurf = state.topg + state.thk

        #state.divflux = tf.where(ACT, state.divflux, 0.0)

def optimize_update_lbfgs(cfg, state, cost, i):

    import tensorflow_probability as tfp

    sc = {}
    sc["thk"] = cfg.processes.data_assimilationCook.scaling.thk
    sc["usurf"] = cfg.processes.data_assimilationCook.scaling.usurf
    sc["slidingco"] = cfg.processes.data_assimilationCook.scaling.slidingco
    sc["arrhenius"] = cfg.processes.data_assimilationCook.scaling.arrhenius

    for f in cfg.processes.data_assimilationCook.control_list:
        vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f])
        if cfg.processes.data_assimilationCook.fitting.log_slidingco & (f == "slidingco"): 
            vars(state)[f+'_sc'] = tf.Variable( tf.sqrt(vars(state)[f] / sc[f]) ) 
        else:
            vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f]) 

    Cost_Glen = []
 
    def COST(controls):

        cost = {}
 
        for i,f in enumerate(cfg.processes.data_assimilationCook.control_list): 
            vars(state)[f+'_sc'] = controls[i]

        for f in cfg.processes.data_assimilationCook.control_list:
            if cfg.processes.data_assimilationCook.fitting.log_slidingco & (f == "slidingco"):
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        update_iceflow_emulated(cfg, state)

        if not cfg.processes.data_assimilationCook.regularization.smooth_anisotropy_factor == 1:
            compute_flow_direction_for_anisotropic_smoothing(state)
                
        return total_cost(cfg, state, cost, i)
        
    def loss_and_gradients_function(controls):
        with tf.GradientTape() as tape:
            tape.watch(controls)
            cost = COST(controls) 
            gradients = tape.gradient(cost, controls)
        return cost, gradients
    
    controls = tf.stack([vars(state)[f+'_sc'] for f in cfg.processes.data_assimilationCook.control_list], axis=0) 
 
    optimizer = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=loss_and_gradients_function,
            initial_position=controls,
            max_iterations=cfg.processes.iceflow.solver.nbitmax,
            tolerance=1e-8)
    
    controls = optimizer.position

    for i,f in enumerate(cfg.processes.data_assimilationCook.control_list):
        vars(state)[f+'_sc'] = controls[i]

    state.divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, 
        method=cfg.processes.data_assimilationCook.divflux.method
    )
    
def update_ncdf_optimize(cfg, state, it):
    """
    Initialize and write the ncdf optimze file
    """

    if hasattr(state, "logger"):
        state.logger.info("Initialize  and write NCDF output Files")
        
    if "velbase_mag" in cfg.processes.data_assimilationCook.output.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in cfg.processes.data_assimilationCook.output.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in cfg.processes.data_assimilationCook.output.vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    
    if "sliding_ratio" in cfg.processes.data_assimilationCook.output.vars_to_save:
        state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    if it == 0:
        nc = Dataset(
            "optimize.nc",
            "w",
            format="NETCDF4",
        )

        nc.createDimension("iterations", None)
        E = nc.createVariable("iterations", np.dtype("float32").char, ("iterations",))
        E.units = "None"
        E.long_name = "iterations"
        E.axis = "ITERATIONS"
        E[0] = it

        nc.createDimension("y", len(state.y))
        E = nc.createVariable("y", np.dtype("float32").char, ("y",))
        E.units = "m"
        E.long_name = "y"
        E.axis = "Y"
        E[:] = state.y.numpy()

        nc.createDimension("x", len(state.x))
        E = nc.createVariable("x", np.dtype("float32").char, ("x",))
        E.units = "m"
        E.long_name = "x"
        E.axis = "X"
        E[:] = state.x.numpy()

        for var in cfg.processes.data_assimilationCook.output.vars_to_save:
            E = nc.createVariable(
                var, np.dtype("float32").char, ("iterations", "y", "x")
            )
            E[0, :, :] = vars(state)[var].numpy()

        nc.close()

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4", )

        d = nc.variables["iterations"][:].shape[0]

        nc.variables["iterations"][d] = it

        for var in cfg.processes.data_assimilationCook.output.vars_to_save:
            nc.variables[var][d, :, :] = vars(state)[var].numpy()

        nc.close()


def output_ncdf_optimize_final(cfg, state):
    """
    Write final geology after optimizing
    """
    if cfg.processes.data_assimilationCook.output.save_iterat_in_ncdf==False:
        if "velbase_mag" in cfg.processes.data_assimilationCook.output.vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if "velsurf_mag" in cfg.processes.data_assimilationCook.output.vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velsurfobs_mag" in cfg.processes.data_assimilationCook.output.vars_to_save:
            state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        
        if "sliding_ratio" in cfg.processes.data_assimilationCook.output.vars_to_save:
            state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    nc = Dataset(
        '../../../Results/RGI-'+cfg.inputs.oggm_shopCook.RGI_ID[-8:-6]+"/"+cfg.inputs.oggm_shopCook.RGI_ID+'_optimised.nc',
        "w",
        format="NETCDF4",
    )

    nc.createDimension("y", len(state.y))
    E = nc.createVariable("y", np.dtype("float32").char, ("y",))
    E.units = "m"
    E.long_name = "y"
    E.axis = "Y"
    E[:] = state.y.numpy()

    nc.createDimension("x", len(state.x))
    E = nc.createVariable("x", np.dtype("float32").char, ("x",))
    E.units = "m"
    E.long_name = "x"
    E.axis = "X"
    E[:] = state.x.numpy()

    for v in cfg.processes.data_assimilationCook.output.vars_to_save:
        if hasattr(state, v):
            E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
            E.standard_name = v
            E[:] = vars(state)[v]

    nc.close()
    
def plot_cost_functions():

#    costs = np.stack(costs)

    file_path = 'costs.dat'

    # Read the file and process the contents
    with open(file_path, 'r') as file:
        lines = file.readlines()
        label = lines[0].strip().split()
        
#        print(lines)
#        print(lines[1:][0].strip().split())
        
        costs = [np.array(line.strip().split(), dtype=float) for line in lines[1:]]
        # costs = [np.array(line.strip().split(), dtype=float) for line in lines[1:]]

    costs = np.stack(costs)

    for i in range(costs.shape[1]):
        costs[:, i] -= np.min(costs[:, i])
        costs[:, i] /= np.where(np.max(costs[:, i]) == 0, 1.0, np.max(costs[:, i]))

    colors = ["k", "r", "b", "g", "c", "m", "k", "r", "b", "g", "c", "m"]
  
    fig = plt.figure(figsize=(10, 10))
    for i in range(costs.shape[1]):
        plt.plot(costs[:, i], label=label[i], c=colors[i])
    plt.ylim(0, 1)
    plt.legend()

    plt.savefig("convergence.png", pad_inches=0)
    plt.close("all")

def update_plot_inversion(cfg, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if cfg.processes.data_assimilationCook.output.editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(2, 3,figsize=(10, 8))

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0, 0]

    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.quantile(state.thk, 0.999),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=12,
    )
    ax1.axis("off")

    #########################################################

    ax2 = state.axes[0, 1]

    from matplotlib import colors

    if "arrhenius" in cfg.processes.data_assimilationCook.control_list:

        im1 = ax2.imshow(
            np.ma.masked_where(state.thk == 0, state.arrhenius),
            origin="lower", 
            vmin=0,
            vmax=300,
            cmap=cmap, 
        )
        if i == 0:
            plt.colorbar(im1, format="%.2f", ax=ax2)
        ax2.set_title("Iteration " + str(i) + " \n Arrhenius coefficient", size=12)
        ax2.axis("off")

    else:

        im1 = ax2.imshow(
            np.ma.masked_where(state.thk == 0, state.slidingco),
            origin="lower",
    #        norm=colors.LogNorm(),
            vmin=0.01,
            vmax=0.10,
            cmap=cmap,
    #        tf.sqrt(state.slidingco/1.0e-6),
    #        vmin=100,
    #        vmax=500,
        )
        if i == 0:
            plt.colorbar(im1, format="%.2f", ax=ax2)
        ax2.set_title("Iteration " + str(i) + " \n Sliding coefficient", size=12)
        ax2.axis("off")

    ########################################################

    ax3 = state.axes[0, 2]

    im1 = ax3.imshow(
        state.usurf - usurfobs,
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax3)
    ax3.set_title(
        "Top surface adjustement \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsusurf[-1], state.stdusurf[-1])
        + ")",
        size=12,
    )
    ax3.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1, 0]

    im1 = ax4.imshow(
        np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=12,
    )
    ax4.axis("off")

    ########################################################

    ax5 = state.axes[1, 1]
    im1 = ax5.imshow(
        np.ma.masked_where(state.thk == 0, velsurfobs_mag),
        origin="lower",
        extent=state.extent,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax5)
    ax5.set_title("Target \n Observed velocities", size=12)
    ax5.axis("off")

    #######################################################

    ax6 = state.axes[1, 2]
    im1 = ax6.imshow(
        state.divflux, # np.where(state.icemaskobs > 0.5, state.divflux,np.nan),
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax6)
    ax6.set_title(
        "Flux divergence \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsdiv[-1], state.stddiv[-1])
        + ")",
        size=12,
    )
    ax6.axis("off")

    #########################################################

    if cfg.processes.data_assimilationCook.output.plot2d_live:
        if cfg.processes.data_assimilationCook.output.editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig("resu-opti-" + str(i).zfill(4) + ".png", bbox_inches="tight", pad_inches=0.2)

def update_plot_inversion_simple(cfg, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if cfg.processes.data_assimilationCook.output.editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(1, 2)

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0]

    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.quantile(state.thk, 0.98),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=16,
    )
    ax1.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1]

    im1 = ax4.imshow(
        np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.nanmax(velsurfobs_mag),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=16,
    )
    ax4.axis("off")

    #########################################################

    if cfg.processes.data_assimilationCook.output.plot2d_live:
        if cfg.processes.data_assimilationCook.output.editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig(
            "resu-opti-" + str(i).zfill(4) + ".png",
            pad_inches=0,
        )
        plt.close("all")

def print_costs(cfg, state, cost, i):

    vol = ( np.sum(state.thk) * (state.dx**2) / 10**9 ).numpy()
    # mean_slidingco = tf.math.reduce_mean(state.slidingco[state.icemaskobs > 0.5])

    f = open('costs.dat','a')

    def bound(x):
        return min(x, 9999999)

    keys = list(cost.keys()) 
    if i == 0:
        L = [f"{key:>8}" for key in ["it","vol"]] + [f"{key:>12}" for key in keys]
        # print("Costs:     " + "   ".join(L))
        print("   ".join([f"{key:>12}" for key in keys]),file=f)

    # if i % cfg.processes.data_assimilationCook.output.freq == 0:
    #     L = [datetime.datetime.now().strftime("%H:%M:%S"),f"{i:0>{8}}",f"{vol:>8.4f}"] \
    #       + [f"{bound(cost[key].numpy()):>12.4f}" for key in keys]
    #     print("   ".join(L))

    print("   ".join([f"{bound(cost[key].numpy()):>12.4f}" for key in keys]),file=f)


def print_info_data_assimilation(cfg, state, cost, i):
    # Compute volume in Gt
    vol = (np.sum(state.thk) * (state.dx**2) / 1e9).numpy()

    # Initialize tqdm bar if needed
    if i % cfg.processes.data_assimilationCook.output.freq == 0:
        if hasattr(state, "pbar_costs"):
            state.pbar_costs.close()
        state.pbar_costs = tqdm(
            desc=" Data assim.", ascii=False, dynamic_ncols=True, bar_format="{desc} {postfix}"
        )

    # Prepare the postfix dictionary
    if hasattr(state, "pbar_costs"):
        dic_postfix = {
            "🕒": datetime.datetime.now().strftime("%H:%M:%S"),
            "🔄": f"{i:04d}",
            "Vol": f"{vol:06.2f}",
        }
        for key in cost:
            value = cost[key].numpy()
            dic_postfix[key] = f"{min(value, 9999999):06.3f}"
        
        state.pbar_costs.set_postfix(dic_postfix)
        state.pbar_costs.update(1)

def save_rms_std(cfg, state):

    np.savetxt(
        "rms_std.dat",
        np.stack(
            [
                state.rmsthk,
                state.stdthk,
                state.rmsvel,
                state.stdvel,
                state.rmsdiv,
                state.stddiv,
                state.rmsusurf,
                state.stdusurf,
            ],
            axis=-1,
        ),
        fmt="%.10f",
        comments='',
        header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
    )

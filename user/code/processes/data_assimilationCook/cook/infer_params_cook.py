import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from igm.processes.utils import getmag
  
def infer_params_cook(state, cfg):
    #This function allows for both parameters to be specified as varying 2D fields (you could compute them pixel-wise from VelMag by swapping in VelMag for VelPerc).
    #This is probably not a good idea, because the values of both parameters do not depend solely on the velocity at that point. But feel free to try! If you do
    #want to do that, you'll also need to un-comment the code block for smoothing and then converting the smoothed weights back to tensors (you may also want to
    #visualise the figures!), and set up a state.convexity_weights field to act as the 2D array
    import scipy
    
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
    TotalArea = TotalArea - (TotalArea*AreaCorrection[RGIRegion]*10)
    TotalVolume = 0.0
    VolBasins = 0.0
    
    state.volumes = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.where(state.icemaskobs > 0, cfg.processes.data_assimilationCook.cook.vol_std, 0.0)
    
    #Get some initial information
    VelMag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    VelMag = tf.where(tf.math.is_nan(VelMag),1e-6,VelMag)
    VelMag = tf.where(VelMag==0,1e-6,VelMag)
    
    #Start of G loop
    for i in range(1,NumGlaciers+1):
        
        #Get area predictor
        Area = tf.reduce_sum(tf.where(state.icemask==i,1.0,0.0))*state.dx**2
        Area = np.round(Area.numpy()/1000**2, decimals=1)
        Area = Area - (Area*AreaCorrection[RGIRegion]*10)
        #print('Area is: ', Area)
        #print('Predicted volume is: ', np.exp(np.log(Area)*1.359622487))
        #Work out nominal volume target based on volume-area scaling - only has much of an effect if no other observations
        if (NumGlaciers/TotalArea < 0.1) & (NumGlaciers > 4) & (TotalArea > 100):
            state.volumes = tf.where(state.icemaskobs > 0.5, 0.034*TotalArea**1.25, state.volumes)
            #print(0.034*TotalArea**1.25)
            if cfg.processes.data_assimilationCook.cook.vol_std == 0.0:
                state.volume_weights = tf.where(state.icemaskobs > 0.5, 0.0, state.volume_weights) #1.0
            else:
                state.volume_weights = tf.where(state.icemaskobs > 0.5, 1.0, state.volume_weights) #1.0
            # cfg.processes.data_assimilationCook.fitting.divfluxobs_std = 10.0
            # cfg.processes.data_assimilationCook.fitting.usurfobs_std = 3.0
            # cfg.processes.data_assimilationCook.regularization.thk = 1.0 #1.0
            #params.opti_velsurfobs_std = 3.0
        else:
            VolBasins = VolBasins + 0.034*Area**1.375
            state.volumes = tf.where(state.icemaskobs > 0.5, VolBasins, state.volumes) #Make sure to put into km3!
        #print('Volume: ',Area,0.034*Area**1.375)
        if Area <= 0.000:
            continue
        
        #Get velocity predictors
        VelMean = np.round(np.mean(VelMag[state.icemaskobs==i]),decimals=2)
        #print("Mean velocity is: ", VelMean)
        
        if VelMean == 0.0:
            #With volume-area scaling (on the assumption these will all be very small)
            state.slidingco = tf.where(state.icemaskobs == i, 0.1, state.slidingco)
            #state.volume_weights = tf.where(state.icemaskobs == i, 0.1, state.volume_weights)
            # state.uvelsurfobs = tf.where(state.icemaskobs == i, np.nan, state.uvelsurfobs)
            # state.vvelsurfobs = tf.where(state.icemaskobs == i, np.nan, state.vvelsurfobs)
            #cfg.processes.data_assimilationCook.fitting.velsurfobs_std = 3.0
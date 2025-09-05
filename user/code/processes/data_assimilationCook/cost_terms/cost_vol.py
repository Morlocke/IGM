#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

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
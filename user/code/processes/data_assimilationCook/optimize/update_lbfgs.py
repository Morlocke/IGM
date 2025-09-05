#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
 
from igm.processes.iceflow.emulate.emulate import update_iceflow_emulated
from igm.processes.utils import compute_divflux
from ..cost_terms.total_cost import total_cost

from ..utils import compute_flow_direction_for_anisotropic_smoothing

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
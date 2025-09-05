#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
 
from igm.processes.iceflow.emulate.emulate import update_iceflow_emulated
from igm.processes.utils import compute_divflux
from ..cost_terms.total_cost import total_cost

from ..utils import compute_flow_direction_for_anisotropic_smoothing

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

        #state.divflux = tf.where(ACT, state.divflux, 0.0)

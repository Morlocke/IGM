# IGM
Weird IGM test cases. Current case is it digging holes in the glacier during an inversion. This may be linked to the use of the volume cost (though the glacier is somewhat below the target volume, so why the algorithm thinks digging a hole would help there is beyond me), the fact that the velocity match seems particularly bad for this glacier (see Franziska's case), or the very small number of NaN velocity pixels (seems unlikely). I have, however, observed it on other glaciers, so I suspect it's the first one, but I can't really see why.

IGM version: 3.0, main branch

igm_run +experiment=params_GlobalInversions_Opti

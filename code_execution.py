from managers import setup_manager,abaqus_manager,results_manager,ai_manager
import numpy as np
from utils import extract_max_strain_rate

def abaqus_process(velocitylist,ratefactorlist):
    analysisdatalist,projectdata=setup_manager.manage_setup(velocitylist,ratefactorlist)
    print('manage setup done')
    abaqus_manager.manage_abaqus(analysisdatalist,projectdata)
    print('manage abaqus done')
    results_manager.manage_postprocess(analysisdatalist,projectdata)
    print('manage postprocess done')
    max_strain_rate = extract_max_strain_rate()
    return max_strain_rate

# velocitylist = [round(velvalue, 0) for velvalue in np.linspace(150,900,10)]
# ratefactorlist = [round(cvalue, 4) for cvalue in np.linspace(0.005,0.06,10)]

def label_inputs(velocitylist, ratefactorlist):
    results = []
    for i in range(len(velocitylist)):
        results.append(abaqus_process([velocitylist[i]],[ratefactorlist[i]]))
    return results
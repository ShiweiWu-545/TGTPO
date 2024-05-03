import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('..')
from models.predictor import DDGPredictor
from models.train import *
from data.my_config import config
from utils.cal_rasa import get_region_ske


if __name__ == '__main__':

    # Fixed random number seed
    set_seed(config.train.seed)

    get_region_ske('WT_1PPF_EI_EI19H.pdb', 'E', 'I', r'D:\Desktop\GToPO\上传\TGTPO\data')

    # data
    batch = load_wt_mut_pdb_pair('../data/WT_1PPF_EI_EI19H.pdb', '../data/MUT_1PPF_EI_EI19H.pdb', '1PPF_EI')
    batch = batch_generator(batch)

    # load model
    weight, _ = load_variable('../data/model_weight.pt')
    model = DDGPredictor(config.model).to(config.model.device)
    model.load_state_dict(load_weight(weight), strict=True)
    model = Optimised_compilation(model)

    # run
    run = run_model()
    yhat = run.make_validation_step(model, batch)
    print('DDG:', yhat)
    print('Positive values indicate a decrease in affinity and negative values indicate an increase in affinity.')
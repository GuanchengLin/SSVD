from datetime import datetime
import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--label_rate", required=True, help="label rate")
parser.add_argument("--dataset", required=True, help="select dataset")
parser.add_argument("--base_model", required=True, help="base model")
parser.add_argument("--cross_val", action='store_true', default=False, help="cross validation")

args = vars(parser.parse_args())
if args['base_model'] == 'LineVul':
    from LineVul_SSVD.HADES_train import *
else:
    from ReVeal_SSVD.HADES_train import *
    
logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

def build_student():

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    with open(f'{args["base_model"]}_SSVD/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    setup_seed(config['train']['random_seed'])

    config['data']['label_rate'] = args['label_rate']
    poolsize = {'big-vul': 30000, 'Juliet': 27000, 'reveal': 16000, 'Devign': 17000}
    config['sampling']['eval_pool_size'] = poolsize[args["dataset"]]
    config['sampling']['sampling_rate'] = 0.4
    config['train']['stopper_mode'] = 'f1acc'
    config['train']['batch_size'] = 16
    config['loss']['coef_student'] = 0.0
    config['sampling']['sampling_scheme'] = 'threshold_confidence'
    config['data']['ssl_data_path'] = f'ssl_data/{args["dataset"]}'

    for slice_num in range(5):
        config['data']['slice'] = slice_num
        job_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        job_name = f'{job_name}_baseline_HADES_{config["loss"]["coef_student"]}{config["loss"]["contrastive_type"]}{config["loss"]["ssvd_trip_alpha"]}_MC_{config["sampling"]["mc_dropout_iters"]}_CL_{config["loss"]["loss_type"]}{config["loss"]["balanced_beta"]}'
        ckpt_path = config['data']['ckpt_path'] + '/' + args["dataset"] + '/' + f'slice_{slice_num}' + '/' + args["label_rate"] + '/' + job_name
        excel_path = config['data']['ckpt_path'] + '/excels/' + args["dataset"] + '/' + f'slice_{slice_num}' + '/' + args["label_rate"]

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        with open(f'{ckpt_path}/config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data=config, stream=f)
        if not os.path.exists(excel_path):
            os.makedirs(excel_path)
        df = pd.DataFrame(columns=['type', 'acc', 'precision', 'recall', 'f1', 'mcc', 'auc', 'kappa', 'tp', 'fp', 'tn', 'fn'])
        excel_path = f'{excel_path}/{job_name}.xlsx'
        df.to_excel(excel_path, index=False, sheet_name='Sheet1')
        
        teacher_path = f'./teachers/{args["dataset"]}/slice_{slice_num}/{args["label_rate"]}/best_teacher.pth'

        try:
            train_HADES(ckpt_path, excel_path, teacher_path, config)
        except Exception as e:
            print("An error occurred:", str(e))
            
        if not args['cross_val']:
            break


if __name__ == '__main__':
    build_student()
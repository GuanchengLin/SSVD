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
    from LineVul_SSVD.train_teacher import *
    from LineVul_SSVD.train_student import *
else:
    from ReVeal_SSVD.train_teacher import *
    from ReVeal_SSVD.train_student import *
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

    config['data']['ssl_data_path'] = f'ssl_data/{args["dataset"]}'
    config['data']['label_rate'] = args['label_rate']
    config['loss']['loss_type'] = 'CE'
    config['loss']['coef_student'] = 0.0
    config['train']['stopper_mode'] = 'f1acc'
    config['sampling']['sampling_scheme'] = 'uniform'
    config['loss']['label_class_loss'] = False
    
    for slice_num in range(5):
        config['data']['slice'] = slice_num
        job_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        job_name = f'{job_name}_baseline_standardST'
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
            train_student(ckpt_path, excel_path, teacher_path, config)
        except Exception as e:
            print("An error occurred:", str(e))
            
        if not args['cross_val']:
            break


if __name__ == '__main__':
    build_student()
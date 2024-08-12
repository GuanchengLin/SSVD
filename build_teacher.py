import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--label_rate", required=True, help="label rate")
parser.add_argument("--dataset", required=True, help="dataset")
parser.add_argument("--stopper_mode", required=True, help="stopper mode")
parser.add_argument("--base_model", required=True, help="base model")
parser.add_argument("--cross_val", action='store_true', default=False, help="cross validation")

args = vars(parser.parse_args())
if args['base_model'] == 'LineVul':
    from LineVul_SSVD.train_teacher import *
else:
    from ReVeal_SSVD.train_teacher import *

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

def build_teacher():

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
    config['data']['ckpt_path'] = f'{args["base_model"]}_SSVD/teachers'
    config['train']['stopper_mode'] = args['stopper_mode']
    config['loss']['loss_type'] = 'CE'
    config['data']['ssl_data_path'] = f'{args["base_model"]}_SSVD/ssl_data/{args["dataset"]}'

    for slice_num in range(5):
        ckpt_path = config['data']['ckpt_path'] + '/' + args["dataset"] + '/' + f'slice_{slice_num}' + '/' + args["label_rate"]
        config['data']['slice'] = slice_num
        
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        with open(f'{ckpt_path}/config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(data=config, stream=f)

        try:
            train_teacher(ckpt_path, config)
        except Exception as e:
            print("An error occurred:", str(e))
        
        if not args['cross_val']:
            break


if __name__ == '__main__':
    build_teacher()
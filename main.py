import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from modules.mlclassifier import GCNClassifier

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr', 'mimic_cxr_2images], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    #edit
    parser.add_argument('--visual_extractor', type=str, default='densenet121', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    #edit
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='seed')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
   
    #KG
    parser.add_argument('--pretrained', type=str, default='models/gcnclassifier_v2_ones3_t401v2t3_lr1e-6_e80.pth', help = 'path of pretrained GCN classifier')
    parser.add_argument('--num_classes', type=int, default=20, help = 'Number of nodes in Knowledge Graph')
    parser.add_argument('--feed_mode', type=str, default = 'both', choices = ['both','cnn_only','gcn_only'], help = 'which features as the input of Transformer')
    
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    
    #edit
    device = torch.device('cuda')
    if args.dataset_name == 'iu_xray':
        fw_adj = torch.tensor([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float,device=device)
    if args.dataset_name == 'mimic_cxr' or 'mimic_cxr_2images':
        fw_adj = torch.tensor([
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] ,
            [0.0, 0.0, 0.4561275751719785, 0.1312437281509043, 0.24146207792929056, 0.5508571762992422, 0.1562972606280271, 0.0, 0.33109282796069434, 0.0, 0.7450978096449961, 0.0, 0.09811571952749415, 0.3566126916516296, 0.4550427125255811, 0.0, 0.0, 0.0, 0.0, 0.3675326731479569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0742334537640819, 0.0, 0.0, 0.04911539003545221] ,
            [0.0, 0.4561275751719785, 0.0, 0.2239353858402398, 0.7210226952471281, 0.1813069680770458, 0.0788364926609991, 0.0, 0.15580397237632496, 0.0, 0.5213886531771248, 0.29374910091081574, 0.07338686671308393, 0.0, 0.47453376787334545, 0.0, 0.0, 0.0, 0.06713445127088644, 0.0, 0.2390801083216384, 0.0, 0.0, 0.5368211205907667, 0.0, 0.12777400002905853, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            [0.0, 0.1312437281509043, 0.2239353858402398, 0.0, 0.6583894493157434, 0.4838989820766221, 0.0, 0.5838389742905232, 0.1802535145956454, 0.0, 0.7991245475975839, 0.11752099418944446, 1.06322936147921, 0.14944553664268462, 0.555751891083365, 0.0, 0.0, 0.0, 0.0, 0.0, 1.194781559134564, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06242749638070836, 0.0, 0.0, 0.0] ,
            [0.0, 0.24146207792929056, 0.7210226952471281, 0.6583894493157434, 0.0, 0.33739430398543896, 0.0, 0.0, 0.2753154828514788, 0.0, 0.7937543770840491, 0.0, 0.44113179799973046, 0.0, 0.3813008244727325, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2757852680983795, 0.0, 0.0, 0.10440811928574718, 0.32527161563918205, 0.30268868564421475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4051634698397344, 0.0, 0.0, 0.0, 0.0, 0.23170964380526302] ,
            [0.0, 0.5508571762992422, 0.1813069680770458, 0.4838989820766221, 0.33739430398543896, 0.0, 0.5084212742711277, 0.6675018415064685, 0.4573178232633596, 0.0, 0.47523016454835715, 0.655060126377754, 0.04554159429343136, 0.690421666641882, 0.558453872850085, 0.0, 0.0, 0.0, 0.0, 0.20852065737883554, 0.6848377491773682, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2185149665526724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04939763863424854] ,
            [0.0, 0.1562972606280271, 0.0788364926609991, 0.0, 0.0, 0.5084212742711277, 0.0, 0.0, 0.0, 0.0, 0.005426229300043639, 1.3057145824116705, 0.0, 0.8349693844237195, 0.008921790018546403, 0.0, 0.0, 0.11331204695360785, 0.09144882429915285, 0.4600716943289044, 0.4835214391246888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02380115748556941, 2.1599410696116648, 0.5656942084851855, 0.0, 0.0, 0.0, 0.5385812163077162, 0.0, 0.0] ,
            [0.0, 0.0, 0.0, 0.5838389742905232, 0.0, 0.6675018415064685, 0.0, 0.0, 0.731672905950786, 0.0, 0.19873089907410066, 1.0717887940778525, 0.4314328875601624, 0.0, 0.0, 0.0, 0.0, 0.0, 0.084615347661925, 0.0, 0.12484394581839403, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13075929436986614, 0.0, 0.08945725659182202, 1.2587049974217708, 0.0, 0.0, 0.0, 0.7231133523008619, 0.24889659448837284, 0.0, 0.06274981602259916] ,
            [0.0, 0.33109282796069434, 0.15580397237632496, 0.1802535145956454, 0.2753154828514788, 0.4573178232633596, 0.0, 0.731672905950786, 0.0, 0.0, 0.3658987804839403, 0.5684867664090824, 0.8076453675499128, 0.0, 0.24600870168688302, 0.0, 0.0, 0.0, 0.0, 0.0798570028567058, 0.6598408690505047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06942284468142029, 0.0, 0.0, 0.0, 0.6390156086235599, 0.0, 0.0, 0.2877393797550929] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37032811695643797, 0.5864507940803438, 0.09648724571559682, 0.08646096743214723, 0.0, 0.0, 0.3795124898565471, 0.5956873219192063, 0.0, 0.0, 0.029427346497570606, 0.3055219304384092, 0.44290494604282604, 0.18031374756716573, 0.0, 0.2817348748514817, 0.0, 0.3037353594240949, 0.0, 0.16902414094509366, 0.44897290997326605, 0.0] ,
            [0.0, 0.7450978096449961, 0.5213886531771248, 0.7991245475975839, 0.7937543770840491, 0.47523016454835715, 0.005426229300043639, 0.19873089907410066, 0.3658987804839403, 0.0, 0.0, 0.15917680219264457, 0.23815070153292747, 0.32301546536533154, 0.4733720905109685, 0.0, 0.0, 0.0, 0.0, 0.3050316392916182, 0.0, 0.0, 0.0, 0.0, 0.007496236022830504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1459218682394831, 0.0, 0.0, 0.0, 0.0, 0.0] ,
            [0.0, 0.0, 0.29374910091081574, 0.11752099418944446, 0.0, 0.655060126377754, 1.3057145824116705, 1.0717887940778525, 0.5684867664090824, 0.0, 0.15917680219264457, 0.0, 0.34332677414193835, 0.2635635043486729, 0.031005417323580087, 0.0, 0.0, 0.0, 0.09089149252550566, 0.02437348630097624, 0.0, 0.0, 0.0, 0.0, 0.03277105749999201, 0.0, 0.0, 0.0, 0.6391742933567508, 0.6037980683638654, 0.0, 0.0, 0.0033714685259266476, 0.23107114275084603, 0.0301477199380383, 0.0, 0.0] ,
            [0.0, 0.09811571952749415, 0.07338686671308393, 1.06322936147921, 0.44113179799973046, 0.04554159429343136, 0.0, 0.4314328875601624, 0.8076453675499128, 0.0, 0.23815070153292747, 0.34332677414193835, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3533266696615023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5094212566296229, 0.0, 0.0, 0.0] ,
            [0.0, 0.3566126916516296, 0.0, 0.14944553664268462, 0.0, 0.690421666641882, 0.8349693844237195, 0.0, 0.0, 0.0, 0.32301546536533154, 0.2635635043486729, 0.0, 0.0, 0.6931424489513275, 0.0, 0.0, 0.0, 0.0, 2.1074196113031913, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15331793735154012, 0.23884807415186537, 0.0, 0.0, 0.0, 0.6251962447501228, 0.9033262867029686, 0.0, 0.0] ,
            [0.0, 0.4550427125255811, 0.47453376787334545, 0.555751891083365, 0.3813008244727325, 0.558453872850085, 0.008921790018546403, 0.0, 0.24600870168688302, 0.0, 0.4733720905109685, 0.031005417323580087, 0.0, 0.6931424489513275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7161636297822183, 0.0, 0.0, 0.0, 0.0, 0.11876223494432343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7766379329419377] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37032811695643797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.826263272280364, 0.572616403287778, 0.37608135779331225, 0.0, 0.0, 0.6664791879765191, 0.0, 0.0, 0.6695968919616625, 0.4488791110312937, 0.38945634739693075, 0.08411274303185438, 0.2917004403335538, 0.3800437779653869, 0.2380677791988365, 0.27679858458872236, 1.9549590442141598, 0.3722905991057948, 0.5307931536036093, 0.7368876097778745, 0.7061522884727808] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5864507940803438, 0.0, 0.0, 0.0, 0.0, 0.0, 2.826263272280364, 0.0, 0.35120934068491455, 0.2375133841107677, 0.0, 0.0, 0.5321065210844121, 0.19927245248782205, 0.0, 0.14957868637482777, 0.0, 0.25536621908603513, 0.0, 0.0, 0.0, 0.0, 0.03734498368480446, 2.516696010229012, 0.0, 0.5307889978371716, 0.0, 0.0960955758632178] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11331204695360785, 0.0, 0.0, 0.09648724571559682, 0.0, 0.0, 0.0, 0.0, 0.0, 0.572616403287778, 0.35120934068491455, 0.0, 0.0, 0.0, 0.0, 0.29348082706206574, 0.33245104866618147, 0.7834253853708174, 0.6637012531860743, 1.0870265661824885, 0.39863505669400867, 0.13312286942605217, 0.4347084018504075, 0.6789980152639449, 1.2829314721133003, 0.8828246864504936, 0.750212390764988, 0.3394040245033589, 0.17754003752713748, 0.3162719096872125, 0.0] ,
            [0.0, 0.0, 0.06713445127088644, 0.0, 0.0, 0.0, 0.09144882429915285, 0.084615347661925, 0.0, 0.08646096743214723, 0.0, 0.09089149252550566, 0.0, 0.0, 0.0, 0.37608135779331225, 0.2375133841107677, 0.0, 0.0, 0.0, 1.6925691084401553, 0.3561732227230664, 0.027819402623596414, 0.7161674728252212, 0.5197157747849462, 0.8581083948136323, 0.42260822276000154, 0.3552253348597958, 0.5997658869837408, 0.720615333108806, 1.319266349117826, 0.5944386313930262, 0.7674724970376314, 0.693554702139206, 0.6387600967878838, 0.4217118279958016, 0.3030211175474031] ,
            [0.0, 0.3675326731479569, 0.0, 0.0, 0.0, 0.20852065737883554, 0.4600716943289044, 0.0, 0.0798570028567058, 0.0, 0.3050316392916182, 0.02437348630097624, 0.0, 2.1074196113031913, 0.7161636297822183, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29382396236601094, 0.0, 0.0, 0.17754549443031645, 0.0, 0.0, 0.32355364182848556, 0.159505799102812, 0.411251947423955, 0.52515309826666, 0.0, 0.596335294846914, 0.013806128482410229, 1.2531067143858092, 1.8230394564602632, 0.13362385380628916, 0.6001536689980314] ,
            [0.0, 0.0, 0.2390801083216384, 1.194781559134564, 0.2757852680983795, 0.6848377491773682, 0.4835214391246888, 0.12484394581839403, 0.6598408690505047, 0.0, 0.0, 0.0, 1.3533266696615023, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6925691084401553, 0.29382396236601094, 0.0, 0.0, 0.0, 0.7138805959551021, 0.11297953392649797, 0.4844330162160724, 0.38596546185743924, 0.036239842558855984, 0.0, 0.5547948133103655, 1.0803235644904228, 0.0, 0.0, 0.7882036912676615, 0.0, 1.240751663702056, 1.0477606549141325] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3795124898565471, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6664791879765191, 0.5321065210844121, 0.29348082706206574, 0.3561732227230664, 0.0, 0.0, 0.0, 0.2534515169611425, 0.0, 0.11246945995471487, 0.07334273346935428, 0.21668590043388075, 0.21941094489706722, 0.9975080075129525, 0.368692533676505, 0.21812689267933633, 0.3287631475798967, 0.7093092033967436, 0.1917101851895807, 0.6056386787874534, 1.1599331154514032, 0.8043554214386265] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5956873219192063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19927245248782205, 0.33245104866618147, 0.027819402623596414, 0.0, 0.0, 0.2534515169611425, 0.0, 0.22725680593423186, 0.0, 0.0, 0.31555285272114897, 0.4267466727339617, 0.0, 0.1826978370690844, 0.0, 0.05379873696374713, 0.14234923917988185, 0.0, 0.08948062941912607, 0.20706340244730248, 0.0] ,
            [0.0, 0.0, 0.5368211205907667, 0.0, 0.10440811928574718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7834253853708174, 0.7161674728252212, 0.17754549443031645, 0.7138805959551021, 0.0, 0.22725680593423186, 0.0, 0.0, 0.4592383775815212, 0.16538956868582963, 0.7629051452564362, 0.0, 0.05346010889771364, 0.0, 0.27509540397847715, 0.21213865741151783, 0.0, 0.22482877173867202, 0.0, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.32527161563918205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007496236022830504, 0.03277105749999201, 0.0, 0.0, 0.11876223494432343, 0.6695968919616625, 0.14957868637482777, 0.6637012531860743, 0.5197157747849462, 0.0, 0.11297953392649797, 0.11246945995471487, 0.0, 0.0, 0.0, 3.2134702267299677, 0.24101556699326157, 0.09145686480864357, 0.12919761705292898, 0.05039604762473017, 0.403553572864001, 1.3069750190832774, 0.7980425960864843, 0.47648926941152736, 1.6040951817884799, 0.0, 0.6259654177637075] ,
            [0.0, 0.0, 0.12777400002905853, 0.0, 0.30268868564421475, 0.0, 0.0, 0.0, 0.0, 0.029427346497570606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4488791110312937, 0.0, 1.0870265661824885, 0.8581083948136323, 0.0, 0.4844330162160724, 0.07334273346935428, 0.0, 0.4592383775815212, 3.2134702267299677, 0.0, 0.2203493961718295, 0.018110233591376573, 0.10255522134088697, 0.26908677536189435, 0.7137634299128567, 1.6439628202548011, 0.925575399120091, 0.4163664995892516, 1.9116571827560045, 0.0, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13075929436986614, 0.0, 0.3055219304384092, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38945634739693075, 0.25536621908603513, 0.39863505669400867, 0.42260822276000154, 0.32355364182848556, 0.38596546185743924, 0.21668590043388075, 0.31555285272114897, 0.16538956868582963, 0.24101556699326157, 0.2203493961718295, 0.0, 0.4371114399604061, 0.5054285531888771, 0.44924515354001515, 0.3648403693789466, 0.28056669723711136, 0.4471627735959794, 0.7437313561632167, 0.4121877509333448, 0.2726999196691929, 0.18922614685864272] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2185149665526724, 0.0, 0.0, 0.0, 0.44290494604282604, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08411274303185438, 0.0, 0.13312286942605217, 0.3552253348597958, 0.159505799102812, 0.036239842558855984, 0.21941094489706722, 0.4267466727339617, 0.7629051452564362, 0.09145686480864357, 0.018110233591376573, 0.4371114399604061, 0.0, 0.20271260668257918, 0.3668239365548465, 0.1184126119193528, 0.5934154784628667, 0.8376418968029923, 0.21619014199138478, 0.28461494476804783, 0.34161510088391334, 1.008623439095296] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02380115748556941, 0.08945725659182202, 0.0, 0.18031374756716573, 0.0, 0.6391742933567508, 0.0, 0.15331793735154012, 0.0, 0.2917004403335538, 0.0, 0.4347084018504075, 0.5997658869837408, 0.411251947423955, 0.0, 0.9975080075129525, 0.0, 0.0, 0.12919761705292898, 0.10255522134088697, 0.5054285531888771, 0.20271260668257918, 0.0, 0.8187047289562672, 0.1410302025894172, 0.3941373798499791, 0.300273095819943, 0.043970911938040466, 0.6307373897434683, 0.09322704591348681, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1599410696116648, 1.2587049974217708, 0.06942284468142029, 0.0, 0.0, 0.6037980683638654, 0.0, 0.23884807415186537, 0.0, 0.3800437779653869, 0.0, 0.6789980152639449, 0.720615333108806, 0.52515309826666, 0.5547948133103655, 0.368692533676505, 0.1826978370690844, 0.05346010889771364, 0.05039604762473017, 0.26908677536189435, 0.44924515354001515, 0.3668239365548465, 0.8187047289562672, 0.0, 1.2341946413520681, 0.35781208803482784, 0.5890899178792679, 0.6340530114404032, 1.0585876788500728, 0.5803311766446675, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5656942084851855, 0.0, 0.0, 0.2817348748514817, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2380677791988365, 0.0, 1.2829314721133003, 1.319266349117826, 0.0, 1.0803235644904228, 0.21812689267933633, 0.0, 0.0, 0.403553572864001, 0.7137634299128567, 0.3648403693789466, 0.1184126119193528, 0.1410302025894172, 1.2341946413520681, 0.0, 1.5406418880016948, 0.7296881633928688, 0.0, 1.8831346562682474, 0.3277653544960015, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.4051634698397344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1459218682394831, 0.0, 0.0, 0.0, 0.0, 0.27679858458872236, 0.03734498368480446, 0.8828246864504936, 0.5944386313930262, 0.596335294846914, 0.0, 0.3287631475798967, 0.05379873696374713, 0.27509540397847715, 1.3069750190832774, 1.6439628202548011, 0.28056669723711136, 0.5934154784628667, 0.3941373798499791, 0.35781208803482784, 1.5406418880016948, 0.0, 2.051483967313616, 0.35982751520577233, 0.8118043924443167, 0.4208658024914072, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3037353594240949, 0.0, 0.0033714685259266476, 0.0, 0.0, 0.0, 1.9549590442141598, 2.516696010229012, 0.750212390764988, 0.7674724970376314, 0.013806128482410229, 0.0, 0.7093092033967436, 0.14234923917988185, 0.21213865741151783, 0.7980425960864843, 0.925575399120091, 0.4471627735959794, 0.8376418968029923, 0.300273095819943, 0.5890899178792679, 0.7296881633928688, 2.051483967313616, 0.0, 0.6607118414843173, 1.251934458983633, 0.6240485023131742, 0.12626765101490742] ,
            [0.0, 0.0742334537640819, 0.0, 0.06242749638070836, 0.0, 0.0, 0.0, 0.7231133523008619, 0.6390156086235599, 0.0, 0.0, 0.23107114275084603, 0.5094212566296229, 0.6251962447501228, 0.0, 0.3722905991057948, 0.0, 0.3394040245033589, 0.693554702139206, 1.2531067143858092, 0.7882036912676615, 0.1917101851895807, 0.0, 0.0, 0.47648926941152736, 0.4163664995892516, 0.7437313561632167, 0.21619014199138478, 0.043970911938040466, 0.6340530114404032, 0.0, 0.35982751520577233, 0.6607118414843173, 0.0, 0.7989276546306372, 0.0, 0.7007917534875767] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5385812163077162, 0.24889659448837284, 0.0, 0.16902414094509366, 0.0, 0.0301477199380383, 0.0, 0.9033262867029686, 0.0, 0.5307931536036093, 0.5307889978371716, 0.17754003752713748, 0.6387600967878838, 1.8230394564602632, 0.0, 0.6056386787874534, 0.08948062941912607, 0.22482877173867202, 1.6040951817884799, 1.9116571827560045, 0.4121877509333448, 0.28461494476804783, 0.6307373897434683, 1.0585876788500728, 1.8831346562682474, 0.8118043924443167, 1.251934458983633, 0.7989276546306372, 0.0, 0.0, 0.0] ,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44897290997326605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7368876097778745, 0.0, 0.3162719096872125, 0.4217118279958016, 0.13362385380628916, 1.240751663702056, 1.1599331154514032, 0.20706340244730248, 0.0, 0.0, 0.0, 0.2726999196691929, 0.34161510088391334, 0.09322704591348681, 0.5803311766446675, 0.3277653544960015, 0.4208658024914072, 0.6240485023131742, 0.0, 0.0, 0.0, 0.0406426973721225] ,
            [0.0, 0.04911539003545221, 0.0, 0.0, 0.23170964380526302, 0.04939763863424854, 0.0, 0.06274981602259916, 0.2877393797550929, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7766379329419377, 0.7061522884727808, 0.0960955758632178, 0.0, 0.3030211175474031, 0.6001536689980314, 1.0477606549141325, 0.8043554214386265, 0.0, 0.0, 0.6259654177637075, 0.0, 0.18922614685864272, 1.008623439095296, 0.0, 0.0, 0.0, 0.0, 0.12626765101490742, 0.7007917534875767, 0.0, 0.0406426973721225, 0.0] ,
    ],dtype = torch.float, device = device)
    
    
    
    bw_adj = fw_adj.t()

    # build model architecture
    submodel = GCNClassifier(args.num_classes, fw_adj, bw_adj) 
    # submodel.state_dict = torch.load(args.pretrained) 


    state_dict = submodel.state_dict()
    state_dict.update({k:v for k, v in torch.load(args.pretrained).items() if k in state_dict})
    submodel.load_state_dict(state_dict)
    
    
    model = R2GenModel(args, tokenizer, submodel)
    # print(model)
    # print(model.state_dict())
    # raise Exception('lol')
    # #edit
    # if args.pretrained:
    #   pretrained_gcn = torch.load(args.pretrained)
    #   pretrained_state_dict = pretrained_gcn['model_state_dict']
    #   state_dict = model.state_dict()
    #   state_dict.update({k: v for k, v in pretrained_state_dict.items() if k in state_dict and 'fc' not in k})
    #   model.load_state_dict(state_dict)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()

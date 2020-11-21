import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', type=str, default='./database')
    parser.add_argument('--dataset', type=str, help='Choose among `Ped2`, `Avenue`,`shanghaitech`', metavar='')
    parser.add_argument('--videoshape', type=int, nargs="+")
    parser.add_argument('--cropshape', type=int, nargs="+")
    # hyperparam
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_flow', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='W', help='weight decay (default: 0.0005)')
    parser.add_argument('--losses', type=str, default=['mse', 'grd'])
    # backbone
    parser.add_argument('--backbone', type=str, default='ITAE')
    parser.add_argument('--step2', type=bool, default=True)
    parser.add_argument('--flow', type=bool, default=True)
    parser.add_argument('--static', type=bool, default=False)
    # flow generative model
    parser.add_argument('--flow_K', type=int, default=32)
    parser.add_argument('--flow_L', type=int, default=3)
    parser.add_argument('--flow_initbatchs', type=int, default=4)
    parser.add_argument('--flow_dyn', type=bool, default=False)
    parser.add_argument('--flow_inputsize', type=int, nargs="+")
    # training
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--SEED', type=int, default=1337)
    parser.add_argument('--GPU', type=str, default='0')
    # save
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_flow_static', type=str, default=None)
    parser.add_argument('--checkpoint_flow_dynamic', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--modelsave', type=str)

    return parser.parse_args()

def arg_txt(args, name=None):
    if not os.path.exists(args.modelsave):
        os.makedirs(args.modelsave)
    if not os.path.exists(args.logsave):
        os.makedirs(args.logsave)
    if name:
        filename = os.path.join(args.modelsave, name)
    else:
        filename = os.path.join(args.modelsave, 'config.txt')

    configtxt = open(filename, 'w')
    configtxt.write(str(args))
    configtxt.close()
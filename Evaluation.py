import torch
import torch.utils.data
import os
from arguments.argument_ITAE import parse_arguments
from datasets.shanghai_dbload import ShanghaiTech
from datasets.Avenue_dbload import Avenue
from datasets.ucsd_ped2_dbload import Ped2
from models.framework import AE
from models.NF import Glow
from utils import set_random_seed
from anomaly_score import VideoAnomalyDetectionResultHelper
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main():
    args = parse_arguments()
    SEED = args.SEED  # random seed for reproduce results
    set_random_seed(SEED)
    DEVICE = torch.device("cuda:%d" % int(args.GPU) if torch.cuda.is_available() else "cpu")
    # =====================================================================================================
    if args.dataset == 'shanghaitech':
        args.videoshape = [3, 16, 260, 260]
        args.cropshape = [3, 16, 256, 256]
        args.flow_inputsize = [32, 32, 2]
        test_dataset = ShanghaiTech(args, train=False)   #dataset load
    elif args.dataset == 'Avenue':
        args.videoshape = [3, 16, 260, 260]
        args.cropshape = [3, 16, 256, 256]
        args.flow_inputsize = [32, 32, 2]
        test_dataset = Avenue(args, train=False)  # dataset load
    elif args.dataset == 'Ped2':
        args.videoshape = [1, 16, 240, 360]
        args.cropshape = [1, 16, 240, 360]
        args.flow_inputsize = [60, 90, 2]
        test_dataset = Ped2(args, train=False)  # dataset load
        args.backbone = args.backbone + '_ped'
    # =====================================================================================================
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)['state_dict']
    else:
        raise NameError('checkpoint ERROR')
    if args.checkpoint_flow_static:
        checkpoint_flow_static = torch.load(args.checkpoint_flow_static, map_location=DEVICE)['state_dict']
    if args.checkpoint_flow_dynamic:
        checkpoint_flow_dynamic = torch.load(args.checkpoint_flow_dynamic, map_location=DEVICE)['state_dict']


    backbone = AE(chnum_in=args.videoshape[0], model=args.backbone)
    if args.flow:
        flow_static = Glow((args.flow_inputsize[0], args.flow_inputsize[1], args.flow_inputsize[2]+1), 512, args.flow_K, args.flow_L, flow_coupling='affine')
        flow_dynamic = Glow((args.flow_inputsize[0], args.flow_inputsize[1], args.flow_inputsize[2]), 512, args.flow_K, args.flow_L, flow_coupling='affine')

    if args.checkpoint:
        backbone.load_state_dict(checkpoint)
        print("Load checkpoint {}".format(args.checkpoint))
        if args.flow:
            flow_static.load_state_dict(checkpoint_flow_static)
            print("Load checkpoint {}".format(args.checkpoint_flow_static))
            flow_dynamic.load_state_dict(checkpoint_flow_dynamic)
            print("Load checkpoint {}".format(args.checkpoint_flow_dynamic))

    args.modelsave = os.path.join(args.save_dir, args.modelsave)
    helper = VideoAnomalyDetectionResultHelper(test_dataset, args)   # for evaluation

    print("Perform Evaluation on "+ str(args.dataset))
    backbone = backbone.to(DEVICE)
    if args.flow:
        flow_static.set_actnorm_init()
        flow_static = flow_static.to(DEVICE)
        flow_dynamic.set_actnorm_init()
        flow_dynamic = flow_dynamic.to(DEVICE)
        helper.test_video_anomaly_detection_multiflow(backbone, flow_static, flow_dynamic, val=False)
    else:
        helper.test_video_anomaly_detection(backbone, flow_static, val=False)

if __name__ == '__main__':
    main()
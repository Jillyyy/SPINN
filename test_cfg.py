import argparse
from config import cfg
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(cfg, args)
# cfg  = '/project/SPINN/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml'

    extra = cfg.MODEL.EXTRA
    print(extra)

if __name__ == '__main__':
    main()
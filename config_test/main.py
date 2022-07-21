# """
# 1引入模块
import argparse
import os.path as osp

from get_config import Config
from utils import mkdir_or_exist
def parse_args():
    ## 这里设置的是 main()当中的 config，即在 Terminal 当中，python 后的参数
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)  ##解析config文件

    ### 保存config到指定目录
    if args.work_dir is not None:  ##在运行代码时，通过 --work-dir 设置保存的目录
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))


    print(cfg)
    print(cfg.square ** 2)

if __name__ == "__main__":
    main()
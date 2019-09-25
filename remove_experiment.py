import argparse
import os

if __name__ == "__main__":
    # Config commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_id', help='experiment id', required=True)
    parser.add_argument('--config_dir', help='configs folder', required=True)
    args = parser.parse_args()
    assert(os.path.isdir(os.path.join(os.getcwd(), 'experiments')))
    mainpath = os.path.join(os.getcwd(), args.config_dir)

    for o in ['configs', 'ckpts', 'logs', 'cached-data']:
        rm_path = os.path.join(mainpath, o)
        cmd = "rm -r {rm_path}/*id_".format(rm_path=rm_path) + str(args.config_id) + "*"
        print(cmd)
        os.system(cmd) 

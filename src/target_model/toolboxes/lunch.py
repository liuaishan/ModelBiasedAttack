import os
import subprocess
import sys
from argparse import ArgumentParser, REMAINDER

if __name__ == '__main__':
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")
    parser.add_argument('training_script_args', nargs=REMAINDER)
    parser.add_argument('--script',
                        metavar="FILE",
                        help="path to script file",
                        type=str)
    parser.add_argument('--count', type=int, default=4)

    current_env = os.environ.copy()
    args = parser.parse_args()

    processes = []
    script = args.script
    for local_rank in range(0, args.count):
        cmd = [sys.executable,
               "-u",
               script,
               "--local_rank={}".format(local_rank),
               "--count={}".format(args.count)] + args.training_script_args
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)

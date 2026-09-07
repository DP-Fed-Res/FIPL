from main import *


gpu_id = 0


if __name__ == '__main__':
    IS_DEBUG = True
    for alg in ['fipl_da']:
        for dataset in ['fl_digits']:
            for gr in [50]:
                for eps in [3.0]:
                    for clip in [2.0]:
                        if IS_DEBUG:
                            debug_command = (f"--model {alg} --dataset {dataset} --communication_epoch {gr} --backbone lightresnet "
                                             f"--local_epoch 1 --online_ratio 1.0  --num_clients 20 "
                                             f"--eps {eps} --max_grad_norm {clip} "
                                             f"--device_id {gpu_id}")
                            print(f">>> {alg}/{dataset}/eps{eps}/C{clip}")
                            args = parse_args(debug_command)
                        else:
                            args = parse_args()
                        main(args)

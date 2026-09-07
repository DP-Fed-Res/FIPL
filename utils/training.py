import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
import os
from collections import Counter
from utils.best_args import dataset_config, get_domain_dict
import time


def get_client_config(args):
    np.random.seed(args.seed)

    num_clients = getattr(args, 'num_clients', 0)
    domain_dict = get_domain_dict(args.dataset, num_clients)

    selected_domain_list = []
    for k, v in domain_dict.items():
        selected_domain_list.extend([k] * v)
    selected_domain_list = np.random.permutation(selected_domain_list).tolist()

    return selected_domain_list, len(selected_domain_list)


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str, args) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace, selected_domain_list=None) -> None:
    domain_names = getattr(private_dataset, 'DOMAINS_LIST', [])
    csv_writer = CsvWriter(args, private_dataset, domain_names=domain_names)

    if hasattr(model, 'set_result_dir'):
        model.set_result_dir(csv_writer.para_foloder_path)

    config = dataset_config.get(args.dataset, {})
    if 'percent_dict' in config:
        private_dataset.percent_dict = config['percent_dict']

    model.N_CLASS = private_dataset.N_CLASS

    print("Running Domain Skew Setting")
    if selected_domain_list is None:
        selected_domain_list, _ = get_client_config(args)

    args.parti_num = len(selected_domain_list)

    result = Counter(selected_domain_list)
    print(result)
    print(selected_domain_list)

    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    model.trainloaders = pri_train_loaders

    client_sample_counts = [len(dl.dataset) for dl in pri_train_loaders]
    domain_sample_counts = {}
    for i, dl in enumerate(pri_train_loaders):
        domain = selected_domain_list[i]
        domain_sample_counts[domain] = domain_sample_counts.get(domain, 0) + len(dl.dataset)
    client_data_info = [
        {'client_id': i, 'domain': selected_domain_list[i], 'sample_count': len(dl.dataset)}
        for i, dl in enumerate(pri_train_loaders)
    ]
    print(f'Client sample counts: {client_sample_counts}')
    print(f'Domain sample counts: {domain_sample_counts}')

    if hasattr(model, 'ini'):
        model.ini()
        model.global_net = model.global_net.to(model.device)

    proto_eps_ratio = getattr(args, 'proto_eps_ratio', 0.0)
    if args.eps > 0:
        has_proto = (model.get_non_model_params_size_per_client() > 0)
        if proto_eps_ratio > 0 and not has_proto:
            print(f'[DP] Warning: proto_eps_ratio={proto_eps_ratio} but model has no prototype upload; '
                  f'ignoring split, using unified mode')
            proto_eps_ratio = 0.0
        if proto_eps_ratio > 1.0 or proto_eps_ratio < 0:
            raise ValueError(f'proto_eps_ratio must be in [0.0, 1.0], got: {proto_eps_ratio}')

        model.dp_enabled = True
        model.dp_total_eps = args.eps
        model.dp_has_proto = has_proto
        model.dp_proto_eps_ratio = proto_eps_ratio
        model.dp_max_grad_norm = args.max_grad_norm

        if proto_eps_ratio > 0:
            eps_proto = args.eps * proto_eps_ratio
            eps_model = args.eps * (1 - proto_eps_ratio)
            print(f'[DP] Explicit split mode: eps={args.eps}, proto_ratio={proto_eps_ratio}')
            print(f'[DP]   eps_model={eps_model:.4f} (DP-SGD), eps_proto={eps_proto:.4f} (prototype)')
            print(f'[DP]   delta={args.delta if hasattr(args, "delta") else 1e-3}, '
                  f'C={args.max_grad_norm}, local_epoch={args.local_epoch}')
            print(f'[DP]   sigma_model and sigma_proto are computed per client based on '
                  f'dataset size and participation rounds')
        else:
            print(f'[DP] Unified mode: eps={args.eps}, delta={args.delta if hasattr(args, "delta") else 1e-3}, '
                  f'C={args.max_grad_norm}, local_epoch={args.local_epoch}')
            print(f'[DP] Unified noise multiplier sigma is computed per client based on '
                  f'local dataset size and participation rounds')
            print(f'[DP] has_proto={has_proto}, prototype steps use q=1.0 in PRV composition')
    else:
        model.dp_enabled = False
        model.dp_total_eps = 0.0
        model.dp_has_proto = False
        model.dp_proto_eps_ratio = 0.0
        model.dp_max_grad_norm = 1.0

    Epoch = args.communication_epoch

    model.build_client_schedule(Epoch)

    accs_dict = {}
    mean_accs_list = []

    model_params_size = model.get_model_params_size()

    online_num = model.online_num
    parti_num = args.parti_num

    model_params_per_round = (online_num + parti_num) * model_params_size
    non_model_bytes_total = 0

    start_time = time.time()

    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index

        if hasattr(model, 'loc_update_GA'):
            model.loc_update_GA(pri_train_loaders, test_loaders, epoch_index)
        elif hasattr(model, 'loc_update'):
            model.loc_update(pri_train_loaders, epoch_index)

        non_model_bytes_total += (
            online_num * model.get_non_model_params_size_per_client()
            + parti_num * model.get_non_model_params_download_size_per_client()
        )

        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME, args)
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)

        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        csv_writer.write_round_acc(epoch_index, accs_dict, mean_acc)

        print('The ' + str(epoch_index) + ' Communication Accuracy:', str(mean_acc), 'Method:', model.args.model)
        print(accs)

    total_time = time.time() - start_time

    non_model_params_per_round = non_model_bytes_total / Epoch

    model.finalize_privacy_accounting()

    best_epoch = int(np.argmax(mean_accs_list))
    best_mean_acc = float(np.max(mean_accs_list))
    final_mean_acc = float(mean_accs_list[-1])

    last5 = mean_accs_list[-5:] if len(mean_accs_list) >= 5 else mean_accs_list
    last5_mean = round(float(np.mean(last5)), 3)
    last5_std = round(float(np.std(last5)), 3)

    per_domain_best = {}
    per_domain_final = {}
    for domain_idx in sorted(accs_dict.keys()):
        domain_accs = accs_dict[domain_idx]
        if domain_idx < len(domain_names):
            label = domain_names[domain_idx]
        else:
            label = str(domain_idx)
        per_domain_best[label] = round(float(np.max(domain_accs)), 2)
        per_domain_final[label] = round(float(domain_accs[-1]), 2)

    client_data_summary = {
        'total_samples': sum(client_sample_counts),
        'per_client_samples': client_sample_counts,
        'per_domain_samples': domain_sample_counts,
        'per_client_detail': client_data_info,
    }

    dp_info = None
    if args.eps > 0:
        has_proto = model.dp_has_proto
        dp_info = {
            'eps': args.eps,
            'eps_target': args.eps,
            'dp_model_sigma': round(model.dp_model_sigma, 4),
            'dp_proto_sigma': round(model.dp_proto_sigma, 4),
            'dp_proto_eps_ratio': model.dp_proto_eps_ratio,
            'is_split_mode': (model.dp_proto_eps_ratio > 0 and has_proto),
            'has_proto': has_proto,
            'max_grad_norm': args.max_grad_norm,
            'delta': args.delta if hasattr(args, 'delta') else 1e-3,
        }
        if dp_info['is_split_mode']:
            dp_info['eps_model_target'] = round(args.eps * (1 - model.dp_proto_eps_ratio), 4)
            dp_info['eps_proto_target'] = round(args.eps * model.dp_proto_eps_ratio, 4)
        if model.client_participation_counts is not None:
            counts = list(model.client_participation_counts.values())
            dp_info['participation_rounds_min'] = min(counts)
            dp_info['participation_rounds_max'] = max(counts)

        if model.client_privacy_consumption:
            dp_info['per_client_privacy'] = {}
            for cid in sorted(model.client_privacy_consumption.keys()):
                entry = dict(model.client_privacy_consumption[cid])
                if cid < len(selected_domain_list):
                    entry['domain'] = selected_domain_list[cid]
                dp_info['per_client_privacy'][str(cid)] = entry

            privacy_lines = []
            privacy_lines.append('=' * 100)
            is_split_mode = model.dp_proto_eps_ratio > 0 and model.dp_has_proto
            mode_label = 'Split accounting' if is_split_mode else 'Unified accounting'
            privacy_lines.append(f'[Privacy Budget Verification] Per-client privacy budget consumption ({mode_label})')
            if is_split_mode:
                privacy_lines.append(
                    f'  Budget split: eps_model={model.dp_total_eps * (1 - model.dp_proto_eps_ratio):.4f}  '
                    f'eps_proto={model.dp_total_eps * model.dp_proto_eps_ratio:.4f}  '
                    f'(ratio={model.dp_proto_eps_ratio})')
            privacy_lines.append('=' * 100)
            if is_split_mode:
                header = (f'{"Client":>8s}  {"Domain":>14s}  {"Target eps":>10s}  {"eps(DP-SGD)":>12s}  '
                          f'{"eps(Proto)":>12s}  {"eps(Total)":>12s}  {"delta":>12s}  '
                          f'{"sigma_m":>10s}  {"sigma_p":>10s}  {"Rounds":>8s}  {"P-Steps":>8s}')
            else:
                header = (f'{"Client":>8s}  {"Domain":>14s}  {"Target eps":>10s}  {"eps(DP-SGD)":>12s}  '
                          f'{"eps(+Protos)":>12s}  {"delta":>12s}  {"sigma":>10s}  '
                          f'{"Rounds":>8s}  {"P-Steps":>8s}')
            privacy_lines.append(header)
            privacy_lines.append('-' * 100)
            for cid in sorted(model.client_privacy_consumption.keys()):
                c = model.client_privacy_consumption[cid]
                total_str = f'{c["epsilon_spent_total"]:>12.6f}' if c['epsilon_spent_total'] is not None else 'N/A'
                domain = selected_domain_list[cid] if cid < len(selected_domain_list) else '?'
                if is_split_mode:
                    proto_eps = c.get('epsilon_spent_proto', 0.0)
                    proto_str = f'{proto_eps:>12.6f}' if isinstance(proto_eps, (int, float)) else f'{proto_eps:>12s}'
                    line = (f'{cid:>8d}  {domain:>14s}  {c["epsilon_target"]:>10.4f}  '
                            f'{c["epsilon_spent_dpsgd"]:>12.6f}  '
                            f'{proto_str:>12s}  '
                            f'{total_str:>12s}  '
                            f'{c["delta"]:>12.2e}  {c["sigma_model"]:>10.4f}  {c["sigma_proto"]:>10.4f}  '
                            f'{c["participation_rounds"]:>8d}  {c["proto_steps"]:>8d}')
                else:
                    line = (f'{cid:>8d}  {domain:>14s}  {c["epsilon_target"]:>10.4f}  '
                            f'{c["epsilon_spent_dpsgd"]:>12.6f}  '
                            f'{total_str:>12s}  '
                            f'{c["delta"]:>12.2e}  {c["sigma"]:>10.4f}  '
                            f'{c["participation_rounds"]:>8d}  {c["proto_steps"]:>8d}')
                privacy_lines.append(line)
            privacy_lines.append('-' * 100)

            dpsgd_values = [v['epsilon_spent_dpsgd'] for v in model.client_privacy_consumption.values()]
            total_values = [v['epsilon_spent_total'] for v in model.client_privacy_consumption.values()
                           if v["epsilon_spent_total"] is not None]
            target_values = [v['epsilon_target'] for v in model.client_privacy_consumption.values()]
            privacy_lines.append(f'eps(DP-SGD only) range: [{min(dpsgd_values):.6f}, {max(dpsgd_values):.6f}]')
            if is_split_mode:
                proto_spent_values = [v.get('epsilon_spent_proto', 0.0) for v in model.client_privacy_consumption.values()]
                privacy_lines.append(f'eps(Protos only) range:  [{min(proto_spent_values):.6f}, {max(proto_spent_values):.6f}]')
            if total_values:
                privacy_lines.append(f'eps(DP-SGD+Proto) range: [{min(total_values):.6f}, {max(total_values):.6f}]')
                privacy_lines.append(f'eps target:               {target_values[0]:.4f} (all clients)')
                privacy_lines.append(f'Max deviation (total):   {max(abs(s - t) for s, t in zip(total_values, target_values)):.6f}')
            privacy_lines.append('=' * 80)

            print()
            for line in privacy_lines:
                print(line)
            print()

            privacy_path = os.path.join(csv_writer.para_foloder_path, 'privacy_budget.txt')
            with open(privacy_path, 'w', encoding='utf-8') as pf:
                pf.write('\n'.join(privacy_lines) + '\n')
            print(f'Privacy budget table saved to: {privacy_path}')

    schedule_info = None
    if model.client_participation_counts is not None:
        counts = list(model.client_participation_counts.values())
        schedule_info = {
            'online_ratio': args.online_ratio,
            'online_num': online_num,
            'participation_rounds_min': min(counts),
            'participation_rounds_max': max(counts),
        }

    summary = {
        'model': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'eps': args.eps,
        'parti_num': parti_num,
        'client_data': client_data_summary,
        'online_num': online_num,
        'communication_epoch': Epoch,
        'local_epoch': args.local_epoch,
        'best_epoch': best_epoch,
        'best_mean_acc': best_mean_acc,
        'final_mean_acc': final_mean_acc,
        'last5_mean_acc': last5_mean,
        'last5_std': last5_std,
        'per_domain_best': per_domain_best,
        'per_domain_final': per_domain_final,
        'avg_comm_per_round_mb': {
            'model_params': round(model_params_per_round / (1024 ** 2), 4),
            'non_model_params': round(non_model_params_per_round / (1024 ** 2), 4),
            'total': round((model_params_per_round + non_model_params_per_round) / (1024 ** 2), 4),
        },
        'total_time_seconds': round(total_time, 2),
        'all_round_mean_acc': mean_accs_list,
        'schedule': schedule_info,
    }
    if dp_info:
        summary['dp'] = dp_info

    csv_writer.write_summary(summary)

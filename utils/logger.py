import copy
import os
import csv
import json
from utils.conf import base_path
from utils.util import create_if_not_exists

useless_args = ['pub_aug', 'public_len', 'public_dataset', 'structure', 'model', 'csv_log',
                'device_id', 'seed', 'tensorboard', 'conf_jobnum', 'conf_timestamp', 'conf_host']


class CsvWriter:
    def __init__(self, args, private_dataset, domain_names=None):
        self.args = args
        self.private_dataset = private_dataset
        self.domain_names = domain_names or []
        self.model_folder_path = self._model_folder_path()
        self.para_foloder_path = self._write_args()
        self._round_accs = {}
        self._mean_accs = []
        print('Results will be saved to: ' + self.para_foloder_path)

    def _model_folder_path(self):
        args = self.args
        data_path = base_path() + args.dataset
        create_if_not_exists(data_path)
        model_path = data_path + '/' + args.model
        create_if_not_exists(model_path)
        return model_path

    def _write_args(self) -> None:
        args = copy.deepcopy(self.args)
        args = vars(args)
        for cc in useless_args:
            if cc in args:
                del args[cc]

        for key, value in args.items():
            args[key] = str(value)

        paragroup_dirs = os.listdir(self.model_folder_path)
        exist_para = False
        path = None

        for para in paragroup_dirs:
            dict_from_csv = {}
            key_value_list = []
            para_path = os.path.join(self.model_folder_path, para)
            args_path = para_path + '/args.csv'
            if not os.path.exists(args_path):
                continue
            try:
                with open(args_path, mode='r') as inp:
                    reader = csv.reader(inp)
                    for rows in reader:
                        key_value_list.append(rows)
                if len(key_value_list) < 2:
                    continue
                for index, _ in enumerate(key_value_list[0]):
                    dict_from_csv[key_value_list[0][index]] = key_value_list[1][index]
            except (IndexError, csv.Error):
                continue
            if args == dict_from_csv:
                path = para_path
                exist_para = True
                break

        if not exist_para:
            n_para = len(paragroup_dirs)
            path = os.path.join(self.model_folder_path, 'para' + str(n_para + 1))
            k = 1
            while os.path.exists(path):
                path = os.path.join(self.model_folder_path, 'para' + str(n_para + k))
                k = k + 1
            create_if_not_exists(path)

            columns = list(args.keys())
            args_path = path + '/args.csv'
            with open(args_path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                writer.writeheader()
                writer.writerow(args)

        return path

    def write_round_acc(self, epoch, accs_dict, mean_acc):
        self._round_accs = accs_dict
        self._mean_accs.append(mean_acc)
        total_epochs = self.args.communication_epoch

        acc_path = os.path.join(self.para_foloder_path, 'acc.csv')
        headers = ['row_label'] + ['epoch_' + str(e) for e in range(total_epochs)]
        with open(acc_path, 'w') as f:
            f.write(','.join(headers) + '\n')
            mean_row = ['mean'] + [str(v) for v in self._mean_accs]
            mean_row += [''] * (total_epochs - len(self._mean_accs))
            f.write(','.join(mean_row) + '\n')
            for domain_idx in sorted(accs_dict.keys()):
                acc_list = accs_dict[domain_idx]
                if domain_idx < len(self.domain_names):
                    label = self.domain_names[domain_idx]
                else:
                    label = f'domain_{domain_idx}'
                row = [label] + [str(v) for v in acc_list]
                row += [''] * (total_epochs - len(acc_list))
                f.write(','.join(row) + '\n')

    def write_summary(self, summary: dict):
        summary_path = os.path.join(self.para_foloder_path, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print('Summary saved to: ' + summary_path)

    def write_loss(self, loss_dict, loss_name):
        import pickle
        loss_path = os.path.join(self.para_foloder_path, loss_name + '.pkl')
        with open(loss_path, 'wb+') as f:
            pickle.dump(loss_dict, f)

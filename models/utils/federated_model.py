import gc
import math

import numpy as np
import torch.nn as nn
import torch
import torchvision
import copy
from argparse import Namespace
from utils.conf import get_device
from utils.dp import add_noise_to_prototype, make_private, find_unified_sigma


class FederatedModel(nn.Module):
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.args = args
        self.transform = transform

        self._model_template = nets_list[0].cpu()
        self.client_params = []
        for net in nets_list:
            self.client_params.append(
                {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
            )
        del nets_list

        # For Online
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.global_params = None
        self.device = get_device(device_id=self.args.device_id)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        self.communication_epoch = args.communication_epoch
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None

        self.epoch_index = 0  # Save the Communication Index

        # DP related parameters (initialized by training.py before training)
        self.dp_enabled = False
        self.dp_total_eps = 0.0
        self.dp_has_proto = False
        self.dp_proto_eps_ratio = 0.0
        self.dp_model_sigma = 0.0
        self.dp_proto_sigma = 0.0
        self.dp_max_grad_norm = 1.0

        # Client participation schedule (built by training.py via build_client_schedule)
        self.client_schedule = None
        self.client_participation_counts = None

        # Per-client privacy budget tracking
        self.client_privacy_consumption = {}
        self._client_delta_cache = {}
        self._client_sigma_cache = {}
        self._client_proto_steps_cache = {}
        self._client_privacy_engine_cache = {}

        # Prototype visualization (shared by subclasses)
        self._result_dir = '.'
        self._viz_rounds = {0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49}

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_net(self):
        return copy.deepcopy(self._model_template)

    @property
    def arch_name(self):
        return self._model_template.name

    # ------------------------------------------------------------------
    # Device / scheduling
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        self.global_net = self._create_net()
        self.global_params = {k: v.clone() for k, v in self.global_net.state_dict().items()}
        for i in range(len(self.client_params)):
            self.client_params[i] = {k: v.clone() for k, v in self.global_params.items()}

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def get_model_params_size(self):
        if self.global_net is not None:
            ref = self.global_net
        else:
            ref = self._model_template
        total = 0
        for p in ref.parameters():
            total += p.numel() * p.element_size()
        return total

    def build_client_schedule(self, total_epochs):
        parti_num = self.args.parti_num
        online_num = self.online_num

        total_slots = online_num * total_epochs
        base_rounds = total_slots // parti_num
        extra_clients = total_slots % parti_num

        self.client_participation_counts = {}
        for cid in range(parti_num):
            if cid < extra_clients:
                self.client_participation_counts[cid] = base_rounds + 1
            else:
                self.client_participation_counts[cid] = base_rounds

        remaining = dict(self.client_participation_counts)
        self.client_schedule = []
        for _ in range(total_epochs):
            available = sorted(
                [(cid, count) for cid, count in remaining.items() if count > 0],
                key=lambda x: (-x[1], self.random_state.random())
            )
            chosen = [cid for cid, _ in available[:online_num]]
            self.random_state.shuffle(chosen)
            self.client_schedule.append(chosen)
            for cid in chosen:
                remaining[cid] -= 1

        online_ratio = online_num / max(parti_num, 1)
        print(f'Client schedule built: {total_epochs} rounds, {online_num}/{parti_num} clients per round '
              f'(online_ratio={online_ratio:.2f})')

    def select_online_clients(self, epoch):
        self.online_clients = self.client_schedule[epoch]
        return self.online_clients

    # ------------------------------------------------------------------
    # DP core methods
    # ------------------------------------------------------------------

    def _make_private(self, net, data_loader, client_id=None):
        if hasattr(data_loader, 'dataset'):
            dataset_size = len(data_loader.dataset)
        else:
            dataset_size = 100  # fallback
        batch_size = data_loader.batch_size if hasattr(data_loader, 'batch_size') else self.args.local_batch_size

        trimmed_size = (dataset_size // batch_size) * batch_size
        if trimmed_size > 0 and trimmed_size < dataset_size:
            from torch.utils.data import Subset, DataLoader
            dataset = Subset(data_loader.dataset, range(trimmed_size))
            data_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     drop_last=True)
            dataset_size = trimmed_size

        if not self.dp_enabled:
            import torch.optim as optim
            optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
            return net, optimizer, data_loader

        if dataset_size < batch_size:
            import torch.optim as optim
            optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
            return net, optimizer, data_loader

        sample_rate = min(batch_size / max(dataset_size, 1), 1.0)
        delta = self.args.delta if hasattr(self.args, 'delta') else 1e-3

        participation_rounds = self.client_participation_counts.get(client_id, 1) if client_id is not None else 1

        if client_id is not None and client_id in self._client_sigma_cache:
            cached = self._client_sigma_cache[client_id]
            if self.dp_proto_eps_ratio > 0 and self.dp_has_proto:
                self.dp_model_sigma, self.dp_proto_sigma = cached
            else:
                self.dp_model_sigma = cached
                self.dp_proto_sigma = cached
        else:
            steps_per_epoch = int(1.0 / sample_rate)
            total_dpsgd_steps = participation_rounds * self.local_epoch * steps_per_epoch
            total_proto_steps = participation_rounds if self.dp_has_proto else 0

            if self.dp_proto_eps_ratio > 0 and self.dp_has_proto:
                from utils.dp import compute_noise_multiplier
                eps_proto = self.dp_total_eps * self.dp_proto_eps_ratio
                eps_model = self.dp_total_eps * (1.0 - self.dp_proto_eps_ratio)

                dp_model_sigma = compute_noise_multiplier(
                    eps_model, delta, sample_rate, total_dpsgd_steps
                )
                dp_proto_sigma = compute_noise_multiplier(
                    eps_proto, delta, 1.0, total_proto_steps
                ) if total_proto_steps > 0 else 0.0

                self.dp_model_sigma = dp_model_sigma
                self.dp_proto_sigma = dp_proto_sigma

                if client_id is not None:
                    self._client_sigma_cache[client_id] = (dp_model_sigma, dp_proto_sigma)
                    self._client_proto_steps_cache[client_id] = total_proto_steps

                print(f'[DP] client={client_id}  eps={self.dp_total_eps}  '
                      f'ratio={self.dp_proto_eps_ratio}  '
                      f'eps_model={eps_model:.4f}  eps_proto={eps_proto:.4f}  '
                      f'sigma_model={dp_model_sigma:.4f}  sigma_proto={dp_proto_sigma:.4f}  '
                      f'C={self.dp_max_grad_norm}  '
                      f'sample_rate={sample_rate:.4f}  '
                      f'dpsgd_steps={total_dpsgd_steps}  proto_steps={total_proto_steps}')
            else:
                history_template = [(sample_rate, total_dpsgd_steps)]
                if total_proto_steps > 0:
                    history_template.append((1.0, total_proto_steps))

                unified_sigma = find_unified_sigma(
                    self.dp_total_eps, delta, history_template
                )
                self.dp_model_sigma = unified_sigma
                self.dp_proto_sigma = unified_sigma

                if client_id is not None:
                    self._client_sigma_cache[client_id] = unified_sigma
                    self._client_proto_steps_cache[client_id] = total_proto_steps

                print(f'[DP] client={client_id}  eps={self.dp_total_eps}  '
                      f'C={self.dp_max_grad_norm}  sigma={unified_sigma:.4f}  '
                      f'sample_rate={sample_rate:.4f}  dpsgd_steps={total_dpsgd_steps}  '
                      f'proto_steps={total_proto_steps}')

        if client_id is not None:
            self._client_delta_cache[client_id] = delta

        cached_engine = self._client_privacy_engine_cache.get(client_id, None) if client_id is not None else None
        net, optimizer, data_loader, privacy_engine = make_private(
            net, data_loader,
            self.dp_model_sigma, self.dp_max_grad_norm,
            lr=self.local_lr, device=str(self.device),
            privacy_engine=cached_engine,
        )
        if client_id is not None and client_id not in self._client_privacy_engine_cache:
            self._client_privacy_engine_cache[client_id] = privacy_engine

        return net, optimizer, data_loader

    # ------------------------------------------------------------------
    # Post-training handling
    # ------------------------------------------------------------------

    def _save_trained_params(self, index, net):
        if hasattr(net, '_module'):
            trained_state = net._module.state_dict()
        else:
            trained_state = net.state_dict()
        self.client_params[index] = {
            k: v.detach().clone().cpu() for k, v in trained_state.items()
        }

    def _clip_feature(self, f):
        if not self.dp_enabled:
            return f
        norm = f.norm(2)
        clip_coef = min(1.0, self.dp_max_grad_norm / (norm + 1e-6))
        return f * clip_coef

    def _add_noise_to_protos(self, protos_dict, sample_counts):
        if not self.dp_enabled or self.dp_proto_sigma <= 0:
            return
        for key in protos_dict:
            n = sample_counts.get(key, 1)
            sensitivity = self.dp_max_grad_norm / max(n, 1)
            val = protos_dict[key]
            if isinstance(val, list):
                protos_dict[key] = [
                    add_noise_to_prototype(t, self.dp_proto_sigma, sensitivity)
                    for t in val
                ]
            else:
                protos_dict[key] = add_noise_to_prototype(
                    val, self.dp_proto_sigma, sensitivity
                )

    def record_privacy_spent(self, client_id, optimizer=None):
        if not self.dp_enabled:
            return

        from utils.dp import get_privacy_spent

        delta = self._client_delta_cache.get(client_id, 1e-3)
        privacy_engine = self._client_privacy_engine_cache.get(client_id, None)
        if privacy_engine is None:
            return

        spent_eps_dpsgd = get_privacy_spent(privacy_engine, delta)

        participation_rounds = (
            self.client_participation_counts.get(client_id, 1)
            if self.client_participation_counts else 1
        )
        proto_steps = self._client_proto_steps_cache.get(client_id, 0)

        is_split = (self.dp_proto_eps_ratio > 0 and self.dp_has_proto)
        sigma_cached = self._client_sigma_cache.get(client_id, self.dp_model_sigma)

        if is_split:
            eps_proto = self.dp_total_eps * self.dp_proto_eps_ratio
            eps_model = self.dp_total_eps * (1.0 - self.dp_proto_eps_ratio)
        else:
            eps_proto = None
            eps_model = None

        self.client_privacy_consumption[client_id] = {
            'epsilon_target': round(self.dp_total_eps, 4),
            'epsilon_spent_dpsgd': round(float(spent_eps_dpsgd), 6),
            'epsilon_spent_total': None,
            'delta': delta,
            'sigma': round(float(sigma_cached[0] if is_split else sigma_cached), 4),
            'sigma_model': round(float(sigma_cached[0] if is_split else sigma_cached), 4),
            'sigma_proto': round(float(sigma_cached[1] if is_split else 0.0), 4),
            'eps_model_target': round(float(eps_model), 4) if eps_model is not None else None,
            'eps_proto_target': round(float(eps_proto), 4) if eps_proto is not None else None,
            'participation_rounds': participation_rounds,
            'proto_steps': proto_steps,
            'is_split': is_split,
        }

    def finalize_privacy_accounting(self):
        if not self.dp_enabled:
            for cid in self.client_privacy_consumption:
                if self.client_privacy_consumption[cid]['epsilon_spent_total'] is None:
                    self.client_privacy_consumption[cid]['epsilon_spent_total'] = \
                        self.client_privacy_consumption[cid]['epsilon_spent_dpsgd']
            return

        if not self.dp_has_proto:
            for cid in self.client_privacy_consumption:
                if self.client_privacy_consumption[cid]['epsilon_spent_total'] is None:
                    self.client_privacy_consumption[cid]['epsilon_spent_total'] = \
                        self.client_privacy_consumption[cid]['epsilon_spent_dpsgd']
            return

        from utils.dp import get_privacy_spent

        is_split = (self.dp_proto_eps_ratio > 0)

        for client_id in list(self.client_privacy_consumption.keys()):
            consumption = self.client_privacy_consumption[client_id]
            if consumption['epsilon_spent_total'] is not None:
                continue

            delta = self._client_delta_cache.get(client_id, 1e-3)
            proto_steps = self._client_proto_steps_cache.get(client_id, 0)
            sigma_cached = self._client_sigma_cache.get(client_id)

            if is_split and proto_steps > 0:
                from opacus.accountants.prv import PRVAccountant
                dpsgd_eps = consumption['epsilon_spent_dpsgd']
                proto_sigma = sigma_cached[1] if isinstance(sigma_cached, tuple) else 0.0

                if proto_sigma > 0:
                    proto_accountant = PRVAccountant()
                    proto_accountant.history = [(proto_sigma, 1.0, proto_steps)]
                    proto_eps = float(proto_accountant.get_epsilon(delta))
                else:
                    proto_eps = 0.0

                consumption['epsilon_spent_proto'] = round(proto_eps, 6)
                consumption['epsilon_spent_total'] = round(float(dpsgd_eps) + proto_eps, 6)
            else:
                privacy_engine = self._client_privacy_engine_cache.get(client_id)
                if privacy_engine is None:
                    consumption['epsilon_spent_total'] = consumption['epsilon_spent_dpsgd']
                    continue

                if proto_steps <= 0:
                    consumption['epsilon_spent_total'] = consumption['epsilon_spent_dpsgd']
                    continue

                sigma = sigma_cached if not isinstance(sigma_cached, tuple) else sigma_cached[0]
                original_history = list(privacy_engine.accountant.history)
                privacy_engine.accountant.history.append((sigma, 1.0, proto_steps))

                spent_eps_total = get_privacy_spent(privacy_engine, delta)
                consumption['epsilon_spent_total'] = round(float(spent_eps_total), 6)

                privacy_engine.accountant.history = original_history

    # ------------------------------------------------------------------
    # Communication cost
    # ------------------------------------------------------------------

    def get_non_model_params_size_per_client(self):
        return 0

    def get_non_model_params_download_size_per_client(self):
        return 0

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _net_features(self, net, x):
        from opacus.grad_sample import GradSampleModule
        if isinstance(net, GradSampleModule):
            return net._module.features(x)
        return net.features(x)

    def _net_classifier(self, net, x):
        from opacus.grad_sample import GradSampleModule
        if isinstance(net, GradSampleModule):
            return net._module.classifier(x)
        return net.classifier(x)

    def set_result_dir(self, path):
        self._result_dir = path

    def get_feature_dim(self):
        net_device = next(self._model_template.parameters()).device
        dummy = torch.randn(1, 3, 224, 224).to(net_device)
        try:
            with torch.no_grad():
                f = self._net_features(self._model_template, dummy)
            return f.shape[1]
        except Exception:
            return 512  # fallback

    # ------------------------------------------------------------------
    # Aggregation & broadcast
    # ------------------------------------------------------------------

    def aggregate_nets(self, freq=None):
        online_clients = self.online_clients

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[idx] for idx in online_clients]
            online_clients_len = [len(dl.dataset) for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        else:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        new_global_params = {}
        first = True
        for idx, client_id in enumerate(online_clients):
            client_params = self.client_params[client_id]
            if first:
                first = False
                for key in client_params:
                    new_global_params[key] = client_params[key].clone() * freq[idx]
            else:
                for key in client_params:
                    new_global_params[key] += client_params[key].clone() * freq[idx]

        self.global_params = new_global_params

        if self.global_net is None:
            self.global_net = self._create_net()
        self.global_net.load_state_dict(self.global_params)
        self.global_net.to(self.device)

        for i in range(len(self.client_params)):
            self.client_params[i] = {k: v.clone() for k, v in self.global_params.items()}

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------

    def _compute_grad_stats(self, net):
        result = {
            'total_norm': 0.0,
            'total_norm_clipped': None,
            'param_mean_norm': 0.0,
            'pct_clipped': None,
            'clipping_ratio': None,
        }

        has_grad_sample = False
        per_sample_norms_all = []
        for p in net.parameters():
            if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                has_grad_sample = True
                gs = p.grad_sample  # shape: [B, *param_shape]
                per_sample_norms = gs.flatten(1).norm(2, dim=1)  # [B]
                per_sample_norms_all.append(per_sample_norms)

        if has_grad_sample and per_sample_norms_all:
            all_norms = torch.cat(per_sample_norms_all)  # [total_params_across_samples]
            all_norms_sq = torch.stack([n ** 2 for n in per_sample_norms_all], dim=0)
            sample_norms = all_norms_sq.sum(dim=0).sqrt()  # [B] per-sample total grad norm
            result['total_norm'] = sample_norms.mean().item()
            result['param_mean_norm'] = all_norms.mean().item()
            result['pct_clipped'] = (sample_norms > self.dp_max_grad_norm).float().mean().item() * 100

            clipped_norms = torch.clamp(sample_norms, max=self.dp_max_grad_norm)
            result['total_norm_clipped'] = clipped_norms.mean().item()
            result['clipping_ratio'] = (clipped_norms.sum() / (sample_norms.sum() + 1e-8)).item()
        else:
            total_norm_sq = 0.0
            n_params = 0
            for p in net.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.norm().item() ** 2
                    n_params += 1
            result['total_norm'] = total_norm_sq ** 0.5
            result['param_mean_norm'] = result['total_norm'] / max(n_params, 1) ** 0.5

        return result

    # ------------------------------------------------------------------
    # Training teardown - release GPU resources
    # ------------------------------------------------------------------

    def _release_training_context(self, net, optimizer, dataloader):
        for p in net.parameters():
            if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                p.grad_sample = None
            if p.grad is not None:
                p.grad = None

        if hasattr(net, 'remove_hooks'):
            try:
                net.remove_hooks()
            except Exception:
                pass

        try:
            net = net.cpu()
        except Exception:
            pass
        if optimizer is not None:
            try:
                for state in optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            state[k] = v.cpu()
            except Exception:
                pass

        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()

        del net, optimizer, dataloader

        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

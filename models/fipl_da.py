import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.args import *
from models.utils.federated_model import FederatedModel


class _CosineClassifier(nn.Linear):
    def __init__(self, in_features, out_features, scale_init=10.0):
        super().__init__(in_features, out_features, bias=False)
        self.scale = nn.Parameter(torch.tensor([scale_init]))
        self.freeze_ce = False

    def forward(self, x):
        x = F.normalize(x, dim=1)
        W = self.weight.detach() if self.freeze_ce else self.weight
        W = F.normalize(W, dim=1)
        return self.scale * (x @ W.t())


def build_etf_classifier(num_classes, feat_dim):
    a = np.random.random(size=(num_classes, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    I = torch.eye(num_classes)
    one = torch.ones(num_classes, num_classes)
    M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(
        P, I - (1.0 / num_classes) * one
    )

    if feat_dim > num_classes:
        r = np.random.random(size=(feat_dim, num_classes))
        Q, _ = np.linalg.qr(r)
        Q = torch.tensor(Q).float()
        W = torch.matmul(M.t(), Q.t())
    elif feat_dim == num_classes:
        W = M.t()
    else:
        W = M.t()[:, :feat_dim]
    return W


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Federated Implicit Prototype Learning (Domain-Aware Alignment).'
    )
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class FIPLDA(FederatedModel):
    NAME = 'fipl_da'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FIPLDA, self).__init__(nets_list, args, transform)

        self.lambda_proto = args.mu if hasattr(args, 'mu') else 1.0

        self.mu_decay = getattr(args, 'mu_decay', 0.9)

        self.etf_lambda = getattr(args, 'etf_lambda', 0.0)

        self.proto_margin = getattr(args, 'proto_margin', 0.0)

        self.etf_lr = getattr(args, 'etf_lr', 0.1)
        self.etf_lr_decay = getattr(args, 'etf_lr_decay', 0.9)

        self.scale_init = getattr(args, 'scale', 10.0)

        self.etf_init = getattr(args, 'etf_init', True)

        self._cls_weight_key = None

    def get_non_model_params_size_per_client(self):
        return 0

    def get_non_model_params_download_size_per_client(self):
        return 0

    def _get_classifier(self, net):
        from opacus.grad_sample import GradSampleModule
        if isinstance(net, GradSampleModule):
            m = net._module
        else:
            m = net

        if hasattr(m, 'linear') and isinstance(m.linear, nn.Linear):
            return m.linear
        if hasattr(m, 'cls') and isinstance(m.cls, nn.Linear):
            return m.cls
        for _, module in reversed(list(m.named_modules())):
            if isinstance(module, nn.Linear):
                return module
        raise AttributeError(
            "Cannot find a Linear classifier in the network. "
            "FIPL-DA requires an nn.Linear classifier layer."
        )

    def _to_cosine_classifier(self, net, scale_init):
        from opacus.grad_sample import GradSampleModule
        cls = self._get_classifier(net)
        new_cls = _CosineClassifier(cls.in_features, cls.out_features, scale_init)
        with torch.no_grad():
            if self.etf_init:
                new_cls.weight.copy_(
                    build_etf_classifier(cls.out_features, cls.in_features)
                )
            else:
                new_cls.weight.copy_(cls.weight)

        m = net._module if isinstance(net, GradSampleModule) else net
        for parent in m.modules():
            for child_name, child in list(parent._modules.items()):
                if child is cls:
                    setattr(parent, child_name, new_cls)

    def _remove_conv4_relu(self, net):
        conv4 = getattr(net, 'conv4', None)
        if not isinstance(conv4, nn.Sequential):
            return
        for i, layer in enumerate(conv4):
            if isinstance(layer, nn.ReLU):
                conv4[i] = nn.Identity()
                return

    def ini(self):
        super().ini()

        for net in [self.global_net, self._model_template]:
            self._to_cosine_classifier(net, self.scale_init)
            cls = self._get_classifier(net)
            cls.freeze_ce = True
            self._remove_conv4_relu(net)

        candidate_keys = ['linear.weight', 'fc.weight', 'classifier.weight', 'cls.weight']
        for key in candidate_keys:
            if key in self.global_params:
                self._cls_weight_key = key
                break
        if self._cls_weight_key is None:
            for key in self.global_params:
                if key.endswith('.weight') and self.global_params[key].ndim == 2:
                    self._cls_weight_key = key
                    break
        if self._cls_weight_key is None:
            raise AttributeError(
                "Cannot locate classifier weight in state_dict keys."
            )

        for net in [self.global_net, self._model_template]:
            if not hasattr(net, 'proto_emb'):
                classifier = self._get_classifier(net)
                C, D = classifier.weight.shape
                emb = nn.Embedding(C, D)
                emb.weight = classifier.weight
                net.add_module('proto_emb', emb)

        self.global_params = {
            k: v.clone() for k, v in self.global_net.state_dict().items()
        }
        for i in range(len(self.client_params)):
            self.client_params[i] = {
                k: v.clone() for k, v in self.global_params.items()
            }

    def loc_update(self, priloader_list, epoch):
        online_clients = self.select_online_clients(epoch)

        for i in online_clients:
            net = self._create_net()
            net.load_state_dict(self.client_params[i])
            self._train_net(i, net, priloader_list[i])

        for i in online_clients:
            self.client_params[i][self._cls_weight_key] = F.normalize(
                self.client_params[i][self._cls_weight_key], dim=1)

        self.aggregate_nets(None)

        self.lambda_proto *= self.mu_decay
        if self.lambda_proto < 5:
            self.lambda_proto = 0

        self.etf_lr *= self.etf_lr_decay

        return None

    def aggregate_nets(self, freq=None):
        online_clients = self.online_clients

        if self.args.averaing == 'weight':
            online_clients_len = [len(self.trainloaders[idx].dataset) for idx in online_clients]
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

        if self.etf_lambda > 0.0 and self._cls_weight_key is not None:
            new_global_params[self._cls_weight_key] = self._solve_etf_prototype(
                online_clients, freq
            )

        self.global_params = new_global_params

        if self.global_net is None:
            self.global_net = self._create_net()
        self.global_net.load_state_dict(self.global_params)
        self.global_net.to(self.device)

        self._log_prototype_diagnostics(online_clients)

        if int(self.epoch_index) in self._viz_rounds:
            self._visualize_prototypes(online_clients)

        for i in range(len(self.client_params)):
            self.client_params[i] = {k: v.clone() for k, v in self.global_params.items()}

    def _solve_etf_prototype(self, online_clients, freq):
        W_mean = None
        for idx, client_id in enumerate(online_clients):
            Wc = self.client_params[client_id][self._cls_weight_key]
            w = 1 / len(online_clients)
            W_mean = Wc * w if W_mean is None else W_mean + Wc * w

        W_mean = F.normalize(W_mean, dim=1)

        W = W_mean.clone()
        steps = getattr(self, 'etf_steps', 20)
        lr = getattr(self, 'etf_lr', 0.1)
        lam = self.etf_lambda
        m = self.proto_margin
        for _ in range(steps):
            S = W @ W.T
            P = F.relu(S - m)
            P.fill_diagonal_(0.0)
            grad = 2.0 * (W - W_mean) + 2.0 * lam * (P @ W)
            W = W - lr * grad
            W = F.normalize(W, dim=1)
        return W

    def _log_prototype_diagnostics(self, online_clients):
        W_global = F.normalize(
            self.global_params[self._cls_weight_key].detach().clone(), dim=1
        )
        K = W_global.shape[0]

        cos_mat = W_global @ W_global.T
        mask = ~torch.eye(K, dtype=torch.bool, device=cos_mat.device)
        off_diag = cos_mat[mask].clamp(-1.0, 1.0)
        angles_deg = torch.rad2deg(torch.acos(off_diag))
        ideal_deg = math.degrees(math.acos(-1.0 / (K - 1))) if K > 1 else float('nan')

        sims = []
        for client_id in online_clients:
            Wc = F.normalize(
                self.client_params[client_id][self._cls_weight_key].detach().clone(),
                dim=1,
            )
            sim = (Wc * W_global).sum(dim=1).mean().item()
            sims.append(sim)

        print(
            f'  [ServerProto] round={self.epoch_index} | '
            f'angle(deg): mean={angles_deg.mean().item():.1f} '
            f'min={angles_deg.min().item():.1f} max={angles_deg.max().item():.1f} '
            f'std={angles_deg.std().item():.1f} (ideal={ideal_deg:.1f}) | '
            f'fit(cos): mean={np.mean(sims):.3f} min={np.min(sims):.3f} max={np.max(sims):.3f} | '
            f'etf_lr={self.etf_lr:.3g}'
        )

    def _visualize_prototypes(self, online_clients):
        from models.utils.prototype_viz import (
            visualize_prototypes, extract_from_classifier_weights)

        W_global = self.global_params[self._cls_weight_key].detach().cpu().numpy()
        K = W_global.shape[0]

        visualize_prototypes(
            W_global,
            list(range(K)),
            [0] * K,
            None,
            self._result_dir, int(self.epoch_index), tag='fipl_da_global')

        W_locals = {
            i: self.client_params[i][self._cls_weight_key].detach()
            for i in online_clients
        }
        vecs, cls, dom, cli = extract_from_classifier_weights(W_locals, online_clients)
        visualize_prototypes(
            vecs, cls, dom, cli,
            self._result_dir, int(self.epoch_index), tag='fipl_da_client')

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)

        net, optimizer, train_loader = self._make_private(
            net, train_loader, client_id=index
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        iterator = tqdm(range(self.local_epoch))
        grad_total_norms = []
        grad_clipped_norms = []
        grad_pct_clipped_list = []

        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)

                f, outputs = net(images, return_features=True)
                f = F.normalize(f, dim=1)

                lossCE = criterion(outputs, labels)

                proto_selected = F.normalize(net.proto_emb(labels), dim=1)
                loss_proto = F.mse_loss(proto_selected, f.detach())

                loss = lossCE + self.lambda_proto * loss_proto
                loss.backward()

                gs = self._compute_grad_stats(net)
                grad_total_norms.append(gs['total_norm'])
                if gs['total_norm_clipped'] is not None:
                    grad_clipped_norms.append(gs['total_norm_clipped'])
                if gs['pct_clipped'] is not None:
                    grad_pct_clipped_list.append(gs['pct_clipped'])

                iterator.desc = (
                    "Local Participant %d CE(w=1)=%.3f Proto(w=%.2g)=%.3f"
                    % (index, lossCE, self.lambda_proto, loss_proto)
                )
                optimizer.step()

        avg_total = sum(grad_total_norms) / max(len(grad_total_norms), 1)
        parts = [f'grad_norm(avg)={avg_total:.4f}']
        if grad_clipped_norms:
            avg_clipped = sum(grad_clipped_norms) / len(grad_clipped_norms)
            avg_pct = sum(grad_pct_clipped_list) / len(grad_pct_clipped_list)
            parts.append(f'clipped={avg_clipped:.4f}')
            parts.append(f'pct_clipped={avg_pct:.1f}%')
            parts.append(f'ratio={avg_clipped / (avg_total + 1e-8):.3f}')
        print(f'  [GradStats] client={index}  ' + '  '.join(parts))

        self._save_trained_params(index, net)
        self.record_privacy_spent(index)
        self._release_training_context(net, optimizer, train_loader)

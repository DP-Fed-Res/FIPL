import os
import numpy as np


def extract_from_classifier_weights(W_locals, online_clients, group_labels=None):
    C = W_locals[online_clients[0]].shape[0]

    vectors = []
    class_labels = []
    domain_labels = []
    client_labels = []

    for ii, i in enumerate(online_clients):
        W = W_locals[i].cpu().numpy()
        for c in range(C):
            vectors.append(W[c, :])
            class_labels.append(c)
            domain_labels.append(int(group_labels[ii]) if group_labels is not None else i)
            client_labels.append(i)

    return np.stack(vectors, axis=0), class_labels, domain_labels, client_labels


def extract_from_proto_dict(proto_dict):
    vectors = []
    class_labels = []
    domain_labels = []

    domain_map = {}  # domain_name -> int
    for key, val in proto_dict.items():
        if isinstance(key, tuple):
            cls_id, domain_name = key
        else:
            cls_id = key
            domain_name = None

        if domain_name is not None:
            if domain_name not in domain_map:
                domain_map[domain_name] = len(domain_map)
            dom_id = domain_map[domain_name]
        else:
            dom_id = 0

        if isinstance(val, list):
            for v in val:
                v_np = v.detach().cpu().numpy().reshape(-1)
                vectors.append(v_np)
                class_labels.append(int(cls_id))
                domain_labels.append(dom_id)
        elif isinstance(val, dict):
            for sub_v in val.values():
                if isinstance(sub_v, list):
                    for sv in sub_v:
                        vectors.append(sv.detach().cpu().numpy().reshape(-1))
                        class_labels.append(int(cls_id))
                        domain_labels.append(dom_id)
                else:
                    vectors.append(sub_v.detach().cpu().numpy().reshape(-1))
                    class_labels.append(int(cls_id))
                    domain_labels.append(dom_id)
        else:
            v_np = val.detach().cpu().numpy().reshape(-1)
            vectors.append(v_np)
            class_labels.append(int(cls_id))
            domain_labels.append(dom_id)

    if len(vectors) == 0:
        return None, None, None

    return np.stack(vectors, axis=0), class_labels, domain_labels


def visualize_prototypes(vectors, class_labels, domain_labels, client_labels,
                         result_dir, epoch, tag='proto', class_names=None):
    if vectors is None or len(vectors) < 2:
        return

    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N = vectors.shape[0]
    C = len(set(class_labels))

    perplexity = min(5, N - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(vectors)

    domain_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                     '#1abc9c', '#e67e22', '#95a5a6', '#34495e', '#d35400']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    unique_domains = sorted(set(domain_labels))
    has_domains = len(unique_domains) > 1
    if has_domains:
        for g in unique_domains:
            mask = np.array(domain_labels) == g
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=domain_colors[g % len(domain_colors)],
                       label=f'Group {g}', s=60, alpha=0.75,
                       edgecolors='#333', linewidth=0.4)
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   c=domain_colors[0], s=60, alpha=0.75,
                   edgecolors='#333', linewidth=0.4)

    if client_labels is not None:
        unique_clients = sorted(set(client_labels))
        for cid in unique_clients:
            mask = np.array(client_labels) == cid
            centroid_2d = X_2d[mask].mean(axis=0)
            ax.annotate(f'c{cid}', xy=centroid_2d, fontsize=9, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    dom_label = f'(K={len(unique_domains)})' if has_domains else '(single group)'
    ax.set_title(f'Round {epoch} - by Domain/Group {dom_label}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    if has_domains:
        ax.legend(fontsize=7)

    ax = axes[1]
    cmap = plt.cm.tab10 if C <= 10 else plt.cm.tab20
    for c in range(C):
        mask = np.array(class_labels) == c
        if class_names and c < len(class_names):
            name = class_names[c]
        else:
            name = f'C{c}'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=cmap(c % 20), label=name, s=60, alpha=0.75,
                   edgecolors='#333', linewidth=0.4)

    ax.set_title(f'Round {epoch} - by Class (C={C})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    if C <= 20:
        ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f'{tag}_tsne_r{epoch:03d}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  [Viz] Prototype t-SNE saved to: {save_path}')


def visualize_prototypes_and_classifiers(proto_vectors, proto_class_labels, proto_client_labels,
                                         clf_vectors, clf_class_labels, clf_client_labels,
                                         result_dir, epoch, tag='fedproto'):
    if proto_vectors is None or clf_vectors is None:
        return
    if len(proto_vectors) < 2 or len(clf_vectors) < 2:
        return

    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    N_proto = proto_vectors.shape[0]
    N_clf = clf_vectors.shape[0]
    C = len(set(proto_class_labels) | set(clf_class_labels))

    all_vectors = np.concatenate([proto_vectors, clf_vectors], axis=0)
    perplexity = min(5, all_vectors.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_all = tsne.fit_transform(all_vectors)

    X_proto = X_all[:N_proto]
    X_clf = X_all[N_proto:]

    cmap = plt.cm.tab10 if C <= 10 else plt.cm.tab20

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    ax = axes[0]
    for c in range(C):
        pmask = np.array(proto_class_labels) == c
        cmask = np.array(clf_class_labels) == c
        color = cmap(c % 20)
        if pmask.any():
            ax.scatter(X_proto[pmask, 0], X_proto[pmask, 1],
                       marker='o', color=color, label=f'C{c} (proto)', s=50, alpha=0.75,
                       edgecolors='#333', linewidth=0.4)
        if cmask.any():
            ax.scatter(X_clf[cmask, 0], X_clf[cmask, 1],
                       marker='x', color=color, label=f'C{c} (clf)', s=70, alpha=0.75,
                       linewidth=1.2)
    ax.set_title(f'Round {epoch} - by Class (o proto, x classifier)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    if C <= 10:
        ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    unique_clients = sorted(set(proto_client_labels) | set(clf_client_labels))
    client_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                     '#1abc9c', '#e67e22', '#95a5a6', '#34495e', '#d35400']
    for client_id in unique_clients:
        color = client_colors[client_id % len(client_colors)]
        pmask = np.array(proto_client_labels) == client_id
        cmask = np.array(clf_client_labels) == client_id
        if pmask.any():
            ax.scatter(X_proto[pmask, 0], X_proto[pmask, 1],
                       marker='o', color=color, label=f'c{client_id} proto', s=50,
                       alpha=0.75, edgecolors='#333', linewidth=0.4)
        if cmask.any():
            centroids = X_clf[cmask].mean(axis=0)
            ax.scatter(X_clf[cmask, 0], X_clf[cmask, 1],
                       marker='x', color=color, label=f'c{client_id} clf', s=70,
                       alpha=0.75, linewidth=1.2)
            ax.annotate(f'c{client_id}', xy=centroids, fontsize=9, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    ax.set_title(f'Round {epoch} - by Client (o proto, x classifier)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    if len(unique_clients) <= 10:
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f'{tag}_tsne_proto_vs_clf_r{epoch:03d}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  [Viz] Proto-vs-Classifier t-SNE saved to: {save_path}')

    fig, axes = plt.subplots(1, len(unique_clients), figsize=(4 * len(unique_clients), 3.5))
    if len(unique_clients) == 1:
        axes = [axes]

    for ax_idx, client_id in enumerate(unique_clients):
        pmask = np.array(proto_client_labels) == client_id

        proto_mat = []  # [present_classes, D]
        clf_mat = []    # [present_classes, D]
        class_list = sorted(set(np.array(proto_class_labels)[pmask]))
        for cl in class_list:
            p_v = proto_vectors[(np.array(proto_class_labels) == cl) &
                                (np.array(proto_client_labels) == client_id)]
            c_v = clf_vectors[(np.array(clf_class_labels) == cl) &
                              (np.array(clf_client_labels) == client_id)]
            if len(p_v) > 0 and len(c_v) > 0:
                proto_mat.append(p_v[0])
                clf_mat.append(c_v[0])

        if len(proto_mat) > 0:
            proto_mat = np.stack(proto_mat, axis=0)  # [K, D]
            clf_mat = np.stack(clf_mat, axis=0)       # [K, D]

            proto_norm = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8)
            clf_norm = clf_mat / (np.linalg.norm(clf_mat, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(proto_norm, clf_norm.T)  # [K, K]

            im = axes[ax_idx].imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
            axes[ax_idx].set_xticks(range(len(class_list)))
            axes[ax_idx].set_xticklabels([f'C{c}' for c in class_list], fontsize=7, rotation=45)
            axes[ax_idx].set_yticks(range(len(class_list)))
            axes[ax_idx].set_yticklabels([f'C{c}' for c in class_list], fontsize=7)
            axes[ax_idx].set_title(f'Client {client_id}')
            axes[ax_idx].set_xlabel('Classifier class')
            axes[ax_idx].set_ylabel('Proto class')

    plt.suptitle(f'Round {epoch} - Cosine Similarity: Proto vs Classifier Weights', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path2 = os.path.join(result_dir, f'{tag}_cosine_sim_r{epoch:03d}.png')
    plt.savefig(save_path2, dpi=150)
    plt.close()
    print(f'  [Viz] Cosine similarity heatmap saved to: {save_path2}')

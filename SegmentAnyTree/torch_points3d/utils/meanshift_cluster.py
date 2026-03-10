import os
from functools import partial

import numpy as np
import torch
from sklearn.cluster import MeanShift
import multiprocessing
def meanshift_cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)  # n_jobs removed in newer sklearn
    #print ('Mean shift clustering, might take some time ...')
    ms.fit(prediction)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_ 	
    #num_clusters = cluster_centers.shape[0]
        
    return torch.from_numpy(labels)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_meanshift_all(all_clusters, fn, processes: int):
    """Run MeanShift over a list of numpy arrays.

    Multiprocessing can easily hang or appear stuck in Docker due to pickle overhead
    (embeddings are large) or Pool worker stalls. Default is to run in-process.

    Controls:
      - MEANSHIFT_USE_MP: enable multiprocessing when >1 processes (default: false)
      - MEANSHIFT_NPROCS: hard cap processes (default: 0 => auto)
    """

    if len(all_clusters) == 0:
        return []

    use_mp = _env_bool("MEANSHIFT_USE_MP", False)
    cap = _env_int("MEANSHIFT_NPROCS", 0)
    if cap > 0:
        processes = min(processes, cap)

    # Never spawn a Pool for single-process work.
    if (not use_mp) or processes <= 1:
        return [fn(arr) for arr in all_clusters]

    # Use a context explicitly; 'fork' is typically fastest on Linux.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        return pool.map(fn, all_clusters, chunksize=1)
        
def cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, low, high, loop_num):
    
    #t = time.time()
    all_clusters = []
    cluster_type = []
    final_result = []
    local_logits = []
    cluster_type_loop = []
    
    embed_logits_logits_u = embed_logits_logits_u.cpu().detach()
    unique_in_batch = unique_in_batch.cpu().detach()
    label_batch = label_batch.cpu().detach()
    local_ind = local_ind.cpu().detach()
    pick_num  = 5 #np.random.randint(low=low,high=high+1,size=loop_num)
    for loop_i in range(loop_num):
        #feature_choose = np.random.choice(embed_logits_logits_u.shape[-1], pick_num[loop_i], replace=False)
        #feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num[loop_i], replacement=False, out=None)
        feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num, replacement=False, out=None)
        #print('type %d feature choose:' % (loop_i))
        #print(feature_choose)
        embed_logits_logits_typei = embed_logits_logits_u[:,feature_choose]
        for s in unique_in_batch:
            batch_mask = label_batch == s
            if torch.sum(batch_mask)>5:
                sampleInBatch_local_ind = local_ind[batch_mask]
                local_logits.append(sampleInBatch_local_ind)
                sample_embed_logits = embed_logits_logits_typei[batch_mask]
                #sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0)
                all_clusters.append(sample_embed_logits.cpu().detach().numpy())
                cluster_type_loop.append(loop_i)
    
    processes = int(unique_in_batch.shape[0]) if unique_in_batch.shape[0] > 0 else 1
    # NOTE: cluster_loop is not used by the default pipeline; keep it safe anyway.
    partial_meanshift_cluster = partial(meanshift_cluster, bandwidth=0.6)
    results = _run_meanshift_all(all_clusters, partial_meanshift_cluster, processes)
    for i in range(len(results)):
        pre_ins_labels_embed = results[i]
        sampleInBatch_local_ind = local_logits[i]
        loop_i_ = cluster_type_loop[i]
        unique_preInslabels = torch.unique(pre_ins_labels_embed)
        for l in unique_preInslabels:
            if l == -1:
                continue
            label_mask_l = pre_ins_labels_embed == l
            final_result.append(sampleInBatch_local_ind[label_mask_l])
            cluster_type.append(loop_i_)
    
    #print("total time",time.time()-t)
    return final_result, cluster_type

def cluster_single(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, type, bandwidth):
    #t = time.time()
    all_clusters = []
    cluster_type = []
    final_result = []
    local_logits = []
    
    embed_logits_logits_u = embed_logits_logits_u.cpu().detach()
    unique_in_batch = unique_in_batch.cpu().detach()
    label_batch = label_batch.cpu().detach()
    local_ind = local_ind.cpu().detach()

    for s in unique_in_batch:
        batch_mask = label_batch == s
        if torch.sum(batch_mask)>3:
            sampleInBatch_local_ind = local_ind[batch_mask]
            local_logits.append(sampleInBatch_local_ind)
            sample_embed_logits = embed_logits_logits_u[batch_mask]
            #meanshift
            #sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0)
            all_clusters.append(sample_embed_logits.cpu().detach().numpy())
            #normalize(sample_embed_logits, axis=0)
    
    partial_meanshift_cluster = partial(meanshift_cluster, bandwidth=bandwidth)
    processes = int(unique_in_batch.shape[0]) if unique_in_batch.shape[0] > 0 else 1

    results = _run_meanshift_all(all_clusters, partial_meanshift_cluster, processes)
    for i in range(len(results)):
        pre_ins_labels_embed = results[i]
        sampleInBatch_local_ind = local_logits[i]
        unique_preInslabels = torch.unique(pre_ins_labels_embed)
        for l in unique_preInslabels:
            if l == -1:
                continue
            label_mask_l = pre_ins_labels_embed == l
            final_result.append(sampleInBatch_local_ind[label_mask_l])
            cluster_type.append(type)
            
            #pre_ins_labels_embed = hdbscan_cluster(sample_embed_logits)
            #unique_preInslabels = torch.unique(pre_ins_labels_embed)
            #for l in unique_preInslabels:
            #    if l == -1:
            #        continue
            #    label_mask_l = pre_ins_labels_embed == l
            #    all_clusters.append(sampleInBatch_local_ind[label_mask_l])
            #    cluster_type.append(type)
                
    #print("total time",time.time()-t)
    return final_result, cluster_type

if __name__ == "__main__":
    # Simple smoke test (optional)
    pass
import torch
import numpy as np
from scipy import stats
from mmengine.logging import MMLogger

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval


@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 binary_mode: str = 'collapsed',
                 binary_ground_eval_id: int = 1,
                 binary_tree_eval_ids=None,
                 report_instance_metrics: bool = True,
                 experiment_tag: str | None = None,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        # Binary mapping controls (eval-id space after _to_eval_sem_ids):
        #   0 = unclassified, 1..K = semantic classes
        # binary_mode:
        #   - 'collapsed': tree = all non-ground classes (defaults to ids 2..K)
        #   - 'strict':    tree = only ids listed in binary_tree_eval_ids
        self.binary_mode = str(binary_mode)
        self.binary_ground_eval_id = int(binary_ground_eval_id)
        self.binary_tree_eval_ids = None if binary_tree_eval_ids is None else list(binary_tree_eval_ids)
        self.report_instance_metrics = bool(report_instance_metrics)
        self.experiment_tag = experiment_tag
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # These are specific to ForAINetV2 evaluation script
        NUM_CLASSES_BINARY = 3  # 0:unclassified, 1:non-tree(stuff), 2:tree(thing)
        num_semantic_classes = len(self.metric_meta.get('classes', [])) if isinstance(self.metric_meta, dict) else 0
        if num_semantic_classes <= 0:
            num_semantic_classes = 3
        # We evaluate with an extra 0="unclassified" bin.
        # Expected eval semantic ids: 0..NUM_CLASSES_SEM-1 where
        # 0=unclassified, 1..num_semantic_classes map to classes.
        NUM_CLASSES_SEM = num_semantic_classes + 1
        INS_CLASS_IDS = [2]  # Instance class is 'wood'/'leaf' which maps to binary 'tree' (2)
        STUFF_CLASS_IDS = [1] # Stuff class is 'ground' which maps to binary 'non-tree' (1)

        def _to_eval_sem_ids(arr: np.ndarray) -> np.ndarray:
            """Normalize dataset-specific semantic ids to eval ids.

            Target eval ids:
              - 0: unclassified/ignored
              - 1..num_semantic_classes: classes in metric_meta['classes'] order

            Observed label conventions in this repo:
              - 0..(K-1)
              - 1..K
              - 2..(K+1)
            """
            a = np.asarray(arr).astype(np.int64)
            k = int(num_semantic_classes)

            # Treat negative as ignore/unclassified.
            valid_nonneg = a[a >= 0]
            if valid_nonneg.size == 0:
                return np.zeros_like(a, dtype=np.int64)

            vmin_nonneg = int(valid_nonneg.min())
            vmax_nonneg = int(valid_nonneg.max())

            # Common convention: 0..K-1 (0 is a valid class) -> +1 (so 0 becomes eval class 1)
            if vmin_nonneg == 0 and vmax_nonneg <= (k - 1):
                return a + 1

            # More general convention: labels are a contiguous positive range with an arbitrary offset,
            # e.g. 1..K, 2..K+1, 3..K+2 (grass/shrub/mallee sometimes appear as 3,4,5).
            valid_pos = valid_nonneg[valid_nonneg > 0]
            if valid_pos.size > 0:
                vmin_pos = int(valid_pos.min())
                vmax_pos = int(valid_pos.max())
                if vmin_pos >= 1 and vmax_pos <= (vmin_pos + k - 1):
                    out = np.zeros_like(a, dtype=np.int64)
                    pos_mask = a > 0
                    out[pos_mask] = a[pos_mask] - (vmin_pos - 1)
                    return out

            # Fallback: keep the previous behavior (+1) and let guards handle out-of-range.
            return a + 1

        def _resolve_to_eval_sem_ids(class_inds) -> list[int]:
            """Convert 0-based class indices into eval semantic ids (1..K).

            In this codepath, semantic eval ids are 1..K, with 0 reserved for unclassified.
            Most configs in this repo pass class indices 0..K-1.
            """
            if class_inds is None:
                return []
            return [int(i) + 1 for i in list(class_inds)]

        # Global accumulators
        true_positive_classes_global = np.zeros(NUM_CLASSES_SEM)
        positive_classes_global = np.zeros(NUM_CLASSES_SEM)
        gt_classes_global = np.zeros(NUM_CLASSES_SEM)
        true_positive_classes_bi_global = np.zeros(NUM_CLASSES_BINARY)
        positive_classes_bi_global = np.zeros(NUM_CLASSES_BINARY)
        gt_classes_bi_global = np.zeros(NUM_CLASSES_BINARY)
        total_gt_ins_global = np.zeros(NUM_CLASSES_BINARY)
        tpsins_global = [[] for _ in range(NUM_CLASSES_BINARY)]
        fpsins_global = [[] for _ in range(NUM_CLASSES_BINARY)]
        IoU_Tp_global = np.zeros(NUM_CLASSES_BINARY)
        all_mean_cov_global = [[] for _ in range(NUM_CLASSES_BINARY)]
        all_mean_weighted_cov_global = [[] for _ in range(NUM_CLASSES_BINARY)]

        # Diagnostics: label histograms after normalization
        sem_gt_hist = np.zeros(NUM_CLASSES_SEM, dtype=np.int64)
        sem_pred_hist = np.zeros(NUM_CLASSES_SEM, dtype=np.int64)
        # Diagnostics: semantic confusion (eval-id space 0..K, where 0=unclassified)
        # Rows = GT, Cols = Pred
        conf_sem = np.zeros((NUM_CLASSES_SEM, NUM_CLASSES_SEM), dtype=np.int64)
        sem_gt_bi_hist = np.zeros(NUM_CLASSES_BINARY, dtype=np.int64)
        sem_pred_bi_hist = np.zeros(NUM_CLASSES_BINARY, dtype=np.int64)

        # Diagnostics: instance-id sanity checks (avoid per-scene spam)
        pred_ins_unique_sample = None
        gt_ins_unique_sample = None
        scenes_with_any_pred_instances = 0
        scenes_with_any_gt_instances = 0
        total_scenes = 0

        # Track unexpected semantic ids (common in zero-shot splits with extra labels)
        remapped_sem_gt = 0
        remapped_sem_pred = 0
        remapped_sem_gt_bi = 0
        remapped_sem_pred_bi = 0

        for eval_ann, single_pred_results in results:
            total_scenes += 1
            # Get GT and Pred labels and normalize to eval ids.
            # (Some splits store labels as 2..4 instead of 0..2, etc.)
            sem_gt_i = _to_eval_sem_ids(eval_ann['pts_semantic_mask'])
            sem_pre_i = _to_eval_sem_ids(single_pred_results['pts_semantic_mask'][1])
            ins_gt_i = eval_ann['pts_instance_mask']
            ins_pre_i = single_pred_results['pts_instance_mask'][1]

            # Guard against off-by-one size mismatch between GT and prediction
            # (can happen due to voxel inverse-mapping rounding)
            n_pts = min(sem_gt_i.shape[0], sem_pre_i.shape[0])
            if sem_gt_i.shape[0] != sem_pre_i.shape[0]:
                import warnings
                warnings.warn(
                    f"Size mismatch: GT={sem_gt_i.shape[0]}, "
                    f"Pred={sem_pre_i.shape[0]}, truncating to {n_pts}"
                )
            sem_gt_i  = sem_gt_i[:n_pts]
            sem_pre_i = sem_pre_i[:n_pts]
            ins_gt_i  = ins_gt_i[:n_pts]
            ins_pre_i = ins_pre_i[:n_pts]

            # Instance-id diagnostics
            # Pred convention in this repo is inconsistent across codepaths;
            # in practice we treat <=0 as background (0) / invalid (-1).
            # GT convention (ForAINetV2): 0 means no instance.
            if pred_ins_unique_sample is None:
                uniq_pred = np.unique(np.asarray(ins_pre_i))
                pred_ins_unique_sample = uniq_pred[:10].tolist()
            if gt_ins_unique_sample is None:
                uniq_gt = np.unique(np.asarray(ins_gt_i))
                gt_ins_unique_sample = uniq_gt[:10].tolist()
            if np.any(np.asarray(ins_pre_i) > 0):
                scenes_with_any_pred_instances += 1
            if np.any(np.asarray(ins_gt_i) != 0):
                scenes_with_any_gt_instances += 1

            # Remap out-of-range semantic ids to 0 (unclassified/ignored)
            # Expected semantic ids: 0..(NUM_CLASSES_SEM-1)
            bad_gt = (sem_gt_i < 0) | (sem_gt_i >= NUM_CLASSES_SEM)
            bad_pred = (sem_pre_i < 0) | (sem_pre_i >= NUM_CLASSES_SEM)
            if np.any(bad_gt):
                remapped_sem_gt += int(np.sum(bad_gt))
                sem_gt_i = sem_gt_i.copy()
                sem_gt_i[bad_gt] = 0
            if np.any(bad_pred):
                remapped_sem_pred += int(np.sum(bad_pred))
                sem_pre_i = sem_pre_i.copy()
                sem_pre_i[bad_pred] = 0
            # Accumulate semantic confusion (fast, vectorized)
            # conf_sem[gt, pred] += count
            conf_sem += np.bincount(
                sem_gt_i * NUM_CLASSES_SEM + sem_pre_i,
                minlength=NUM_CLASSES_SEM * NUM_CLASSES_SEM
            ).reshape(NUM_CLASSES_SEM, NUM_CLASSES_SEM)

            # Semantic Segmentation Evaluation
            sem_gt_hist[: len(np.bincount(sem_gt_i, minlength=NUM_CLASSES_SEM))] += np.bincount(
                sem_gt_i, minlength=NUM_CLASSES_SEM
            )
            sem_pred_hist[: len(np.bincount(sem_pre_i, minlength=NUM_CLASSES_SEM))] += np.bincount(
                sem_pre_i, minlength=NUM_CLASSES_SEM
            )
            for j in range(n_pts):
                gt_l, pred_l = int(sem_gt_i[j]), int(sem_pre_i[j])
                # gt_l/pred_l are now guaranteed within [0, NUM_CLASSES_SEM)
                gt_classes_global[gt_l] += 1
                positive_classes_global[pred_l] += 1
                true_positive_classes_global[gt_l] += int(gt_l == pred_l)

            # Binary Semantic and Instance Evaluation
            # Binary mapping in eval-id space:
            #   0 = unclassified
            #   1 = non-tree (ground/background)
            #   2 = tree
            sem_gt_bi = np.zeros_like(sem_gt_i, dtype=np.int64)
            sem_pre_bi = np.zeros_like(sem_pre_i, dtype=np.int64)

            ground_eval_id = int(self.binary_ground_eval_id)
            if self.binary_tree_eval_ids is not None:
                tree_eval_ids = np.array([int(x) for x in self.binary_tree_eval_ids], dtype=np.int64)
            else:
                # Default: everything except ground (ids 2..K) counts as tree.
                tree_eval_ids = np.arange(2, NUM_CLASSES_SEM, dtype=np.int64)

            # Non-tree (ground)
            sem_gt_bi[sem_gt_i == ground_eval_id] = 1
            sem_pre_bi[sem_pre_i == ground_eval_id] = 1

            mode = self.binary_mode.lower().strip()
            if mode in ('collapsed', 'default', 'non_ground'):
                # Tree = any semantic class not equal to ground (typically wood/leaf, etc.)
                sem_gt_bi[np.isin(sem_gt_i, tree_eval_ids)] = 2
                sem_pre_bi[np.isin(sem_pre_i, tree_eval_ids)] = 2
            elif mode in ('strict', 'mallee_only', 'tree_only'):
                # Tree = exactly the ids listed in binary_tree_eval_ids
                sem_gt_bi[np.isin(sem_gt_i, tree_eval_ids)] = 2
                sem_pre_bi[np.isin(sem_pre_i, tree_eval_ids)] = 2
            else:
                logger.warning(
                    f"UnifiedSegMetric: unknown binary_mode={self.binary_mode!r}; falling back to 'collapsed'."
                )
                sem_gt_bi[np.isin(sem_gt_i, tree_eval_ids)] = 2
                sem_pre_bi[np.isin(sem_pre_i, tree_eval_ids)] = 2

            # Any remaining labels (e.g. extra categories) are treated as unclassified (0)
            bad_gt_bi = (sem_gt_bi < 0) | (sem_gt_bi >= NUM_CLASSES_BINARY)
            bad_pred_bi = (sem_pre_bi < 0) | (sem_pre_bi >= NUM_CLASSES_BINARY)
            if np.any(bad_gt_bi):
                remapped_sem_gt_bi += int(np.sum(bad_gt_bi))
                sem_gt_bi = sem_gt_bi.copy()
                sem_gt_bi[bad_gt_bi] = 0
            if np.any(bad_pred_bi):
                remapped_sem_pred_bi += int(np.sum(bad_pred_bi))
                sem_pre_bi = sem_pre_bi.copy()
                sem_pre_bi[bad_pred_bi] = 0

            for j in range(sem_gt_bi.shape[0]):
                gt_l, pred_l = int(sem_gt_bi[j]), int(sem_pre_bi[j])
                gt_classes_bi_global[gt_l] += 1
                positive_classes_bi_global[pred_l] += 1
                true_positive_classes_bi_global[gt_l] += int(gt_l == pred_l)

            sem_gt_bi_hist[: len(np.bincount(sem_gt_bi, minlength=NUM_CLASSES_BINARY))] += np.bincount(
                sem_gt_bi, minlength=NUM_CLASSES_BINARY
            )
            sem_pred_bi_hist[: len(np.bincount(sem_pre_bi, minlength=NUM_CLASSES_BINARY))] += np.bincount(
                sem_pre_bi, minlength=NUM_CLASSES_BINARY
            )

            if not self.report_instance_metrics:
                continue

            # Filter out points that are ground in both pred and gt for instance evaluation
            idxc = (sem_gt_bi != 1) | (sem_pre_bi != 1)
            pred_ins, gt_ins = ins_pre_i[idxc], ins_gt_i[idxc]
            pred_sem, gt_sem = sem_pre_bi[idxc], sem_gt_bi[idxc]

            # Get predicted instances
            un = np.unique(pred_ins)
            pts_in_pred = [[] for _ in range(NUM_CLASSES_BINARY)]
            for g in un:
                # Skip background / invalid ids.
                if g <= 0:
                    continue
                tmp = (pred_ins == g)
                sem_seg_i = int(stats.mode(pred_sem[tmp], keepdims=True)[0][0])
                if 0 <= sem_seg_i < NUM_CLASSES_BINARY:
                    pts_in_pred[sem_seg_i].append(tmp)

            # Get ground truth instances
            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES_BINARY)]
            for g in un:
                if g == 0: continue # In ForAINetV2, instance ID 0 is not a valid instance
                tmp = (gt_ins == g)
                sem_seg_i = int(stats.mode(gt_sem[tmp], keepdims=True)[0][0])
                if 0 <= sem_seg_i < NUM_CLASSES_BINARY:
                    pts_in_gt[sem_seg_i].append(tmp)

            # Coverage Metrics (MUCov, MWCov)
            for i_sem in INS_CLASS_IDS:
                if not pts_in_gt[i_sem] or not pts_in_pred[i_sem]: continue
                sum_cov, num_gt_point, mean_weighted_cov = 0, 0, 0
                for ins_gt in pts_in_gt[i_sem]:
                    ovmax = 0.
                    num_ins_gt_point = np.sum(ins_gt)
                    num_gt_point += num_ins_gt_point
                    for ins_pred in pts_in_pred[i_sem]:
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)
                        if iou > ovmax: ovmax = iou
                    sum_cov += ovmax
                    mean_weighted_cov += ovmax * num_ins_gt_point
                if len(pts_in_gt[i_sem]) != 0:
                    all_mean_cov_global[i_sem].append(sum_cov / len(pts_in_gt[i_sem]))
                    all_mean_weighted_cov_global[i_sem].append(mean_weighted_cov / num_gt_point)

            # PQ, SQ, RQ Metrics
            for i_sem in INS_CLASS_IDS:
                tp, fp = [0.] * len(pts_in_pred[i_sem]), [0.] * len(pts_in_pred[i_sem])
                IoU_Tp_per = 0
                if pts_in_gt[i_sem]: total_gt_ins_global[i_sem] += len(pts_in_gt[i_sem])

                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.
                    if not pts_in_gt[i_sem]:
                        fp[ip] = 1; continue
                    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)
                        if iou > ovmax: ovmax = iou
                    if ovmax >= 0.5:
                        tp[ip] = 1
                        IoU_Tp_per += ovmax
                    else:
                        fp[ip] = 1
                tpsins_global[i_sem].extend(tp)
                fpsins_global[i_sem].extend(fp)
                IoU_Tp_global[i_sem] += IoU_Tp_per

        # Final Metric Calculation
        metrics = dict()

        # Semantic Segmentation
        iou_list = []
        valid_sem_classes = [i for i, n in enumerate(gt_classes_global) if n > 0 and i > 0] # Exclude unclassified
        for i in range(1, NUM_CLASSES_SEM):
            iou = true_positive_classes_global[i] / float(gt_classes_global[i] + positive_classes_global[i] - true_positive_classes_global[i] + 1e-8)
            iou_list.append(iou)
        metrics['mIoU'] = np.mean([iou_list[i-1] for i in valid_sem_classes]) if valid_sem_classes else 0.0

        # Per-class IoU (semantic)
        class_names = []
        if isinstance(self.metric_meta, dict):
            class_names = list(self.metric_meta.get('classes', []))
        if not class_names:
            class_names = [f'class{i}' for i in range(1, NUM_CLASSES_SEM)]
        for sem_id in range(1, NUM_CLASSES_SEM):
            name = class_names[sem_id - 1] if (sem_id - 1) < len(class_names) else f'class{sem_id}'
            metrics[f'IoU_{name}'] = float(iou_list[sem_id - 1])

        # Binary Semantic Segmentation
        iou_list_bi = []
        valid_bi_sem_classes = [i for i, n in enumerate(gt_classes_bi_global) if n > 0 and i > 0]
        for i in range(1, NUM_CLASSES_BINARY):
            iou = true_positive_classes_bi_global[i] / float(gt_classes_bi_global[i] + positive_classes_bi_global[i] - true_positive_classes_bi_global[i] + 1e-8)
            iou_list_bi.append(iou)
        metrics['mIoU_binary'] = np.mean([iou_list_bi[i-1] for i in valid_bi_sem_classes]) if valid_bi_sem_classes else 0.0

        # Per-class IoU (binary)
        # 1=non-tree(ground), 2=tree(wood+leaf)
        if len(iou_list_bi) >= 1:
            metrics['IoU_non_tree'] = float(iou_list_bi[0])
        if len(iou_list_bi) >= 2:
            metrics['IoU_tree'] = float(iou_list_bi[1])

        if self.report_instance_metrics:
            # Instance Segmentation
            MUCov = np.zeros(NUM_CLASSES_BINARY)
            MWCov = np.zeros(NUM_CLASSES_BINARY)
            precision = np.zeros(NUM_CLASSES_BINARY)
            recall = np.zeros(NUM_CLASSES_BINARY)
            RQ = np.zeros(NUM_CLASSES_BINARY)
            SQ = np.zeros(NUM_CLASSES_BINARY)
            PQ = np.zeros(NUM_CLASSES_BINARY)
            for i_sem in INS_CLASS_IDS:
                MUCov[i_sem] = np.mean(all_mean_cov_global[i_sem]) if all_mean_cov_global[i_sem] else 0
                MWCov[i_sem] = np.mean(all_mean_weighted_cov_global[i_sem]) if all_mean_weighted_cov_global[i_sem] else 0
                tp = np.sum(tpsins_global[i_sem])
                fp = np.sum(fpsins_global[i_sem])
                rec = tp / (total_gt_ins_global[i_sem] + 1e-8)
                prec = tp / (tp + fp + 1e-8)
                precision[i_sem], recall[i_sem] = prec, rec
                RQ[i_sem] = 2 * prec * rec / (prec + rec + 1e-8)
                SQ[i_sem] = IoU_Tp_global[i_sem] / (tp + 1e-8)
                PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
            valid_ins_classes = [i for i in INS_CLASS_IDS if total_gt_ins_global[i] > 0]
            if not valid_ins_classes:
                valid_ins_classes = INS_CLASS_IDS  # Avoid division by zero if no GT

            metrics['mMWCov'] = np.mean(MWCov[valid_ins_classes])
            metrics['mMUCov'] = np.mean(MUCov[valid_ins_classes])
            metrics['mPrecision'] = np.mean(precision[valid_ins_classes])
            metrics['mRecall'] = np.mean(recall[valid_ins_classes])
            metrics['F1'] = (2 * metrics['mPrecision'] * metrics['mRecall']) / (
                metrics['mPrecision'] + metrics['mRecall'] + 1e-8
            )
            metrics['mPQ'] = np.mean(PQ[valid_ins_classes])
            metrics['mSQ'] = np.mean(SQ[valid_ins_classes])
            metrics['mRQ'] = np.mean(RQ[valid_ins_classes])

        # Balanced semantic-instance score for checkpoint selection.
        # harmonic = 2 * mIoU * F1 / (mIoU + F1)
        if 'mIoU' in metrics and 'F1' in metrics:
            metrics['harmonic'] = (2 * metrics['mIoU'] * metrics['F1']) / (
                metrics['mIoU'] + metrics['F1'] + 1e-8
            )

        log_str = 'Evaluation Results:\n'
        if self.experiment_tag:
            log_str += f"Experiment: {self.experiment_tag}\n"
        log_str += f"mIoU: {metrics['mIoU']:.4f}, mIoU_binary: {metrics['mIoU_binary']:.4f}\n"
        # Per-class semantic IoUs (dynamic based on metric_meta['classes']).
        sem_classes = []
        if isinstance(self.metric_meta, dict):
            sem_classes = list(self.metric_meta.get('classes', []))
        if not sem_classes:
            sem_classes = [f'class{i}' for i in range(1, NUM_CLASSES_SEM)]

        iou_parts = []
        for name in sem_classes[: (NUM_CLASSES_SEM - 1)]:
            iou_parts.append(f"IoU_{name}: {metrics.get(f'IoU_{name}', 0.0):.4f}")
        log_str += ", ".join(iou_parts) + "\n"
        if self.report_instance_metrics:
            log_str += f"mPQ: {metrics['mPQ']:.4f}, mSQ: {metrics['mSQ']:.4f}, mRQ: {metrics['mRQ']:.4f}\n"
            log_str += (
                f"mPrecision: {metrics['mPrecision']:.4f}, mRecall: {metrics['mRecall']:.4f}, "
                f"F1: {metrics['F1']:.4f}\n"
            )
            log_str += f"mMUCov: {metrics['mMUCov']:.4f}, mMWCov: {metrics['mMWCov']:.4f}"
        if 'harmonic' in metrics:
            log_str += f"\nHarmonic(mIoU,F1): {metrics['harmonic']:.4f}"

        # Extra diagnostics: GT/pred label distributions
        class_names = []
        if isinstance(self.metric_meta, dict):
            class_names = list(self.metric_meta.get('classes', []))
        if not class_names:
            class_names = [f'class{i}' for i in range(1, NUM_CLASSES_SEM)]

        diag_lines = ["\n[diag] Semantic label counts (eval ids)"]
        diag_lines.append(f"  GT   : {sem_gt_hist.tolist()}")
        diag_lines.append(f"  Pred : {sem_pred_hist.tolist()}")
        diag_lines.append("  (id 0=unclassified, ids 1..K=" + ",".join(class_names) + ")")
        diag_lines.append("[diag] Binary label counts (0=unclassified,1=non-tree,2=tree)")
        diag_lines.append(f"  GT   : {sem_gt_bi_hist.tolist()}")
        diag_lines.append(f"  Pred : {sem_pred_bi_hist.tolist()}")
        diag_lines.append("[diag] Instance-id sanity")
        diag_lines.append(
            f"  Scenes with any GT instances   : {scenes_with_any_gt_instances}/{total_scenes}"
        )
        diag_lines.append(
            f"  Scenes with any Pred instances : {scenes_with_any_pred_instances}/{total_scenes}"
        )
        if gt_ins_unique_sample is not None:
            diag_lines.append(f"  GT unique ids sample   : {gt_ins_unique_sample} (0 means background)")
        if pred_ins_unique_sample is not None:
            diag_lines.append(f"  Pred unique ids sample : {pred_ins_unique_sample} (<=0 treated as background/invalid)")
        # --- Shrub confusion summary (works for any class names order) ---
        # eval ids are: 0=unclassified, 1..K correspond to metric_meta['classes'] order
        classes = class_names  # you already built this above from metric_meta

        def _eval_id_for(name: str) -> int | None:
            try:
                return classes.index(name) + 1  # +1 because eval-id starts at 1
            except ValueError:
                return None

        id_hummock = _eval_id_for('hummock')
        id_shrub   = _eval_id_for('shrub')
        id_mallee  = _eval_id_for('mallee')

        if id_shrub is not None:
            gt_shrub_total_all = int(conf_sem[id_shrub, :].sum())
            gt_shrub_total_inK = int(conf_sem[id_shrub, 1:].sum())

            to_0 = int(conf_sem[id_shrub, 0])
            to_h = int(conf_sem[id_shrub, id_hummock]) if id_hummock is not None else 0
            to_s = int(conf_sem[id_shrub, id_shrub])
            to_m = int(conf_sem[id_shrub, id_mallee]) if id_mallee is not None else 0

            diag_lines.append("\n[diag] Shrub confusion (GT=shrub)")
            diag_lines.append(f"  GT shrub total (all preds 0..K): {gt_shrub_total_all}")
            diag_lines.append(f"  GT shrub total (pred in 1..K): {gt_shrub_total_inK}")
            diag_lines.append(f"  shrub→unclassified(0): {to_0}")
            diag_lines.append(f"  shrub→hummock: {to_h}")
            diag_lines.append(f"  shrub→shrub  : {to_s}")
            diag_lines.append(f"  shrub→mallee : {to_m}")

            den_all = max(gt_shrub_total_all, 1)
            diag_lines.append(
                f"  pct_all: to_0={to_0/den_all*100:.1f}% "
                f"to_hummock={to_h/den_all*100:.1f}% "
                f"to_shrub={to_s/den_all*100:.1f}% "
                f"to_mallee={to_m/den_all*100:.1f}%"
            )

            den_inK = max(gt_shrub_total_inK, 1)
            diag_lines.append(
                f"  pct_inK: to_hummock={to_h/den_inK*100:.1f}% "
                f"to_shrub={to_s/den_inK*100:.1f}% "
                f"to_mallee={to_m/den_inK*100:.1f}%"
            )
        log_str += "\n" + "\n".join(diag_lines)

        if remapped_sem_gt or remapped_sem_pred or remapped_sem_gt_bi or remapped_sem_pred_bi:
            log_str += (
                "\n[warn] Remapped out-of-range semantic ids to 0 (unclassified): "
                f"sem_gt={remapped_sem_gt}, sem_pred={remapped_sem_pred}, "
                f"sem_gt_bi={remapped_sem_gt_bi}, sem_pred_bi={remapped_sem_pred_bi}"
            )

        if self.report_instance_metrics:
            # Paper-style row (percent) matching Table 1 columns, but with dynamic
            # semantic class names (so medium/low splits don’t incorrectly show
            # Ground/Wood/Leaf).
            sem_cols = sem_classes[: (NUM_CLASSES_SEM - 1)]
            sem_header = " ".join(sem_cols)
            sem_vals = " ".join(
                f"{metrics.get(f'IoU_{name}', 0.0) * 100:.1f}" for name in sem_cols
            )
            log_str += (
                f"\nTable1Row(%) [Prec Rec F1 Cov {sem_header} mIoU]: "
                f"{metrics['mPrecision'] * 100:.1f} "
                f"{metrics['mRecall'] * 100:.1f} "
                f"{metrics['F1'] * 100:.1f} "
                f"{metrics['mMUCov'] * 100:.1f} "
                f"{sem_vals} "
                f"{metrics['mIoU'] * 100:.1f}"
            )

        if self.report_instance_metrics:
            # Stage 1 — sanity-check line matching common benchmark expectations:
            # Semantic: Ground vs Tree (Tree = Wood + Leaf)
            # Instance: individual tree masks (Prec/Rec/F1/Cov)
            log_str += (
                "\nStage1Row(%) [Prec Rec F1 Cov Ground Tree mIoU]: "
                f"{metrics['mPrecision'] * 100:.1f} "
                f"{metrics['mRecall'] * 100:.1f} "
                f"{metrics['F1'] * 100:.1f} "
                f"{metrics['mMWCov'] * 100:.1f} "
                f"{metrics.get('IoU_non_tree', 0.0) * 100:.1f} "
                f"{metrics.get('IoU_tree', 0.0) * 100:.1f} "
                f"{metrics['mIoU_binary'] * 100:.1f}"
            )

        # Experiment 1 — zero-shot transfer limits:
        # Binary semantic evaluation only (Non-veg vs Veg).
        # Non-veg = ground, Veg = wood + leaf.
        log_str += (
            "\nExp1Row(%) [NonVeg Veg mIoU]: "
            f"{metrics.get('IoU_non_tree', 0.0) * 100:.1f} "
            f"{metrics.get('IoU_tree', 0.0) * 100:.1f} "
            f"{metrics['mIoU_binary'] * 100:.1f}"
        )
        logger.info(log_str)
        
        return metrics


@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(
                single_pred_results['pts_instance_mask'])
            pred_instance_labels.append(single_pred_results['instance_labels'])
            pred_instance_scores.append(single_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict
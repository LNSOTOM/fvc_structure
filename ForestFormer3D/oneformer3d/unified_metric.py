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
        NUM_CLASSES_BINARY = 3  # unclassified, non-tree, tree
        NUM_CLASSES_SEM = 3  # After remapping: 0=hummock, 1=shrub, 2=mallee
        INS_CLASS_IDS = [2]  # Instance class is mallee (2 after remap) which maps to binary 'tree' (2)
        STUFF_CLASS_IDS = [0, 1] # Stuff classes are hummock(0) and shrub(1) which map to binary 'non-tree' (1)
        
        # Semantic label remapping: actual TEST data {1,2,3} → model expects {0,1,2}
        # Test data classes (preprocessed .bin files):
        #   1 = hummock (stuff, no instances)  → 0 (stuff)
        #   2 = shrub (stuff, no instances)    → 1 (stuff) 
        #   3 = mallee (thing, HAS instances)  → 2 (thing)
        SEMANTIC_REMAP = {1: 0, 2: 1, 3: 2}

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

        for eval_ann, single_pred_results in results:
            # Get GT and Pred labels
            sem_gt_raw = eval_ann['pts_semantic_mask'].copy()
            ins_gt_raw = eval_ann['pts_instance_mask']
            
            # Remap semantic labels: {1,2,3} → {0,1,2}
            sem_gt_i = sem_gt_raw.copy()
            for old_cls, new_cls in SEMANTIC_REMAP.items():
                sem_gt_i[sem_gt_raw == old_cls] = new_cls
            
            logger.info(f"GT original classes: {np.unique(sem_gt_raw)} → remapped: {np.unique(sem_gt_i)}")
            
            # Check which GT class has instances (should be class 2 = mallee after remap)
            for cls_id in np.unique(sem_gt_i):
                n_instances = len(np.unique(ins_gt_raw[sem_gt_i == cls_id])) - 1  # -1 for instance 0
                if n_instances > 0:
                    logger.info(f"GT class {cls_id} has {n_instances} instances (expected: class 2=mallee)")
            
            # Predictions are already in model space {0,1,2} (ground, wood, leaf)
            # No remapping needed for predictions
            sem_pre_voxel = single_pred_results['pts_semantic_mask'][0]
            ins_gt_i = ins_gt_raw
            
            # Convert boolean instance mask to integer instance IDs
            # pts_instance_mask[0] is a boolean array (n_points, n_instances)
            ins_pred_mask_voxel = single_pred_results['pts_instance_mask'][0]
            if ins_pred_mask_voxel.ndim == 2:  # Boolean mask format (n_points, n_instances)
                # Check if transposed - if first dim is small, it's (n_instances, n_points)
                if ins_pred_mask_voxel.shape[0] < ins_pred_mask_voxel.shape[1]:
                    ins_pred_mask_voxel = ins_pred_mask_voxel.T
                # Convert to integer instance IDs
                ins_pre_voxel = np.full(ins_pred_mask_voxel.shape[0], -1, dtype=np.int32)
                for inst_id in range(ins_pred_mask_voxel.shape[1]):
                    ins_pre_voxel[ins_pred_mask_voxel[:, inst_id]] = inst_id
            else:  # Already in integer format
                ins_pre_voxel = ins_pred_mask_voxel
            
            # FIXED: Evaluate on voxelized points only (where predictions exist)
            # originids contains indices of original points that were voxelized
            if 'originids' in single_pred_results:
                originids = single_pred_results['originids']
                logger.info(f"Evaluating on voxelized points: {originids.shape[0]} / {sem_gt_i.shape[0]} ({100*originids.shape[0]/sem_gt_i.shape[0]:.1f}%)")
                
                # Subset GT to match voxelized predictions
                sem_gt_i = sem_gt_i[originids]
                ins_gt_i = ins_gt_i[originids]
                sem_pre_i = sem_pre_voxel
                ins_pre_i = ins_pre_voxel
                
                logger.info(f"After subsetting - GT:{sem_gt_i.shape[0]}, Pred:{sem_pre_i.shape[0]}")
            else:
                logger.warning("originids not found! Falling back to truncation")
                sem_pre_i = sem_pre_voxel
                ins_pre_i = ins_pre_voxel
                
                min_size = min(sem_gt_i.shape[0], sem_pre_i.shape[0], ins_gt_i.shape[0], ins_pre_i.shape[0])
                logger.info(f"Truncating to {min_size} points")
                sem_gt_i = sem_gt_i[:min_size]
                sem_pre_i = sem_pre_i[:min_size]
                ins_gt_i = ins_gt_i[:min_size]
                ins_pre_i = ins_pre_i[:min_size]
            
            min_size = sem_gt_i.shape[0]

            # Semantic Segmentation Evaluation
            for j in range(min_size):
                gt_l, pred_l = int(sem_gt_i[j]), int(sem_pre_i[j])
                gt_classes_global[gt_l] += 1
                positive_classes_global[pred_l] += 1
                true_positive_classes_global[gt_l] += int(gt_l == pred_l)

            # Binary Semantic and Instance Evaluation
            # Map semantic labels to binary: 0 for unclassified, 1 for non-tree (stuff), 2 for tree (thing)
            sem_gt_bi = np.ones_like(sem_gt_i)  # Default to non-tree (1)
            sem_pre_bi = np.ones_like(sem_pre_i)  # Default to non-tree (1)
            
            # For GT after remapping: class 0=hummock, class 1=shrub (stuff), class 2=mallee (thing with instances)
            sem_gt_bi[sem_gt_i == 0] = 1  # hummock → non-tree (stuff)
            sem_gt_bi[sem_gt_i == 1] = 1  # shrub → non-tree (stuff)
            sem_gt_bi[sem_gt_i == 2] = 2  # mallee → tree (thing)
            
            # For predictions in this project: class 0=hummock (stuff), class 1=shrub (stuff), class 2=mallee (thing)
            sem_pre_bi[sem_pre_i == 0] = 1  # hummock → non-tree (stuff)
            sem_pre_bi[sem_pre_i == 1] = 1  # shrub → non-tree (stuff)
            sem_pre_bi[sem_pre_i == 2] = 2  # mallee → tree (thing)
            
            logger.info(f"Binary mapping - GT: {np.sum(sem_gt_bi == 2)} tree points, Pred: {np.sum(sem_pre_bi == 2)} tree points")

            # Binary semantic evaluation
            for j in range(min_size):
                gt_l, pred_l = int(sem_gt_bi[j]), int(sem_pre_bi[j])
                gt_classes_bi_global[gt_l] += 1
                positive_classes_bi_global[pred_l] += 1
                true_positive_classes_bi_global[gt_l] += int(gt_l == pred_l)

            # Filter out points that are ground in both pred and gt for instance evaluation
            idxc = (sem_gt_bi != 1) | (sem_pre_bi != 1)
            pred_ins, gt_ins = ins_pre_i[idxc], ins_gt_i[idxc]
            pred_sem, gt_sem = sem_pre_bi[idxc], sem_gt_bi[idxc]

            # Get predicted instances
            un = np.unique(pred_ins)
            pts_in_pred = [[] for _ in range(NUM_CLASSES_BINARY)]
            for g in un:
                if g == -1: continue
                tmp = (pred_ins == g)
                sem_seg_i = int(stats.mode(pred_sem[tmp], keepdims=True)[0][0])
                pts_in_pred[sem_seg_i].append(tmp)

            # Get ground truth instances
            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES_BINARY)]
            for g in un:
                if g == 0: continue # In ForAINetV2, instance ID 0 is not a valid instance
                tmp = (gt_ins == g)
                sem_seg_i = int(stats.mode(gt_sem[tmp], keepdims=True)[0][0])
                pts_in_gt[sem_seg_i].append(tmp)
                if g == 1:  # Debug first GT instance
                    logger.info(f"First GT instance semantic class: {sem_seg_i}, unique sem values: {np.unique(gt_sem[tmp])}")
            
            # Debug logging
            logger.info(f"Sample: {len(pts_in_pred[2])} predicted tree instances, {len(pts_in_gt[2])} GT tree instances, {len(pts_in_gt[1])} GT non-tree")

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
        class_names = self.metric_meta.get('classes', ['hummock', 'shrub', 'mallee'])
        valid_sem_classes = [i for i, n in enumerate(gt_classes_global) if n > 0 and i < NUM_CLASSES_SEM]
        
        logger.info("\nPer-class Semantic IoU:")
        for i in range(NUM_CLASSES_SEM):
            iou = true_positive_classes_global[i] / float(gt_classes_global[i] + positive_classes_global[i] - true_positive_classes_global[i] + 1e-8)
            iou_list.append(iou)
            if i < len(class_names):
                logger.info(f"  Class {i} ({class_names[i]}): IoU={iou:.4f}, GT={int(gt_classes_global[i])}, Pred={int(positive_classes_global[i])}, TP={int(true_positive_classes_global[i])}")
        
        metrics['mIoU'] = np.mean([iou_list[i] for i in valid_sem_classes]) if valid_sem_classes else 0.0

        # Binary Semantic Segmentation
        iou_list_bi = []
        binary_names = ['non-tree', 'tree']
        valid_bi_sem_classes = [i for i, n in enumerate(gt_classes_bi_global) if n > 0 and i > 0]
        
        logger.info("\nBinary Semantic IoU:")
        for i in range(1, NUM_CLASSES_BINARY):
            iou = true_positive_classes_bi_global[i] / float(gt_classes_bi_global[i] + positive_classes_bi_global[i] - true_positive_classes_bi_global[i] + 1e-8)
            iou_list_bi.append(iou)
            logger.info(f"  Binary class {i} ({binary_names[i-1]}): IoU={iou:.4f}, GT={int(gt_classes_bi_global[i])}, Pred={int(positive_classes_bi_global[i])}, TP={int(true_positive_classes_bi_global[i])}")
        
        metrics['mIoU_binary'] = np.mean([iou_list_bi[i-1] for i in valid_bi_sem_classes]) if valid_bi_sem_classes else 0.0

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
        if not valid_ins_classes: valid_ins_classes = INS_CLASS_IDS # Avoid division by zero if no GT

        metrics['mMWCov'] = np.mean(MWCov[valid_ins_classes])
        metrics['mMUCov'] = np.mean(MUCov[valid_ins_classes])
        metrics['mPrecision'] = np.mean(precision[valid_ins_classes])
        metrics['mRecall'] = np.mean(recall[valid_ins_classes])
        metrics['F1'] = (2 * metrics['mPrecision'] * metrics['mRecall']) / (metrics['mPrecision'] + metrics['mRecall'] + 1e-8)
        metrics['mPQ'] = np.mean(PQ[valid_ins_classes])
        metrics['mSQ'] = np.mean(SQ[valid_ins_classes])
        metrics['mRQ'] = np.mean(RQ[valid_ins_classes])

        # Compact percent summaries for quick reporting
        tree_class_id = INS_CLASS_IDS[0] if INS_CLASS_IDS else 2
        logger.info("\nIndividual Tree Seg. (%):")
        logger.info(
            "  "
            f"Prec={precision[tree_class_id] * 100:.2f}  "
            f"Rec={recall[tree_class_id] * 100:.2f}  "
            f"F1={RQ[tree_class_id] * 100:.2f}  "
            f"Cov(MWCov)={MWCov[tree_class_id] * 100:.2f}"
        )

        logger.info("\nSemantic Seg. (%):")
        for i in range(NUM_CLASSES_SEM):
            name = class_names[i] if i < len(class_names) else str(i)
            logger.info(f"  {name}: IoU={iou_list[i] * 100:.2f}")
        logger.info(f"  mIoU: {metrics['mIoU'] * 100:.2f}")

        logger.info(f"\nInstance Metrics Summary:")
        logger.info(f"  Total GT instances: {int(total_gt_ins_global[2])}")
        logger.info(f"  Total predicted instances: {len(tpsins_global[2])}")
        logger.info(f"  True positives (IoU≥0.5): {int(np.sum(tpsins_global[2]))}")
        logger.info(f"  False positives: {int(np.sum(fpsins_global[2]))}")

        log_str = '\nEvaluation Results:\n'
        log_str += f"mIoU: {metrics['mIoU']:.4f}, mIoU_binary: {metrics['mIoU_binary']:.4f}\n"
        log_str += f"mPQ: {metrics['mPQ']:.4f}, mSQ: {metrics['mSQ']:.4f}, mRQ: {metrics['mRQ']:.4f}\n"
        log_str += f"mPrecision: {metrics['mPrecision']:.4f}, mRecall: {metrics['mRecall']:.4f}, F1: {metrics['F1']:.4f}\n"
        log_str += f"mMUCov: {metrics['mMUCov']:.4f}, mMWCov: {metrics['mMWCov']:.4f}"
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

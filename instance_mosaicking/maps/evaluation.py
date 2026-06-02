import os
import numpy as np
import geopandas as gpd

def IoU(poly1, poly2):
    i = poly1.intersection(poly2).area
    u = poly1.union(poly2).area
    return i/u

def uniqueness(TP, TN):
    a = TP + TN
    unq, unq_idx, unq_cnt = np.unique(a, return_inverse=True, return_counts=True)
    cnt_mask = unq_cnt > 1
    dup_ids = unq[cnt_mask]
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx = np.argsort(unq_idx[idx_mask])
    dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
    duplicated_ids = [a[i[0]] for i in dup_idx if i.size !=0]
    return duplicated_ids

def evaluate(groundtruth_shapefile, merged_shapefile, IoU_threshold=0.88):
    assert os.path.exists(groundtruth_shapefile)
    assert os.path.exists(merged_shapefile)
    gt_shp = gpd.read_file(groundtruth_shapefile)
    merged_shp = gpd.read_file(merged_shapefile)
    N, _ = merged_shp.shape
    TP = []
    FP = []
    TN = []
    FN = []
    for i in range(N):
        merged_poly = merged_shp['geometry'][i]
        ids = merged_shp['id'][i]
        if ',' in ids:
            # merged result: postive
            ids = [int(id) for id in ids.split(',')]
            if np.unique(ids).shape[0] == 1:
                id = ids[0]
                gt_poly = gt_shp['geometry'][id]
                if IoU(gt_poly, merged_poly) > IoU_threshold:
                    # true positive
                    TP.append(id)
                else:
                    # false positive
                    FP.append(id)
            else:
                # false positive
                FP.append(ids)
        else:
            # unmerged result: negative
            id = int(ids)
            gt_poly = gt_shp['geometry'][id]
            if IoU(gt_poly, merged_poly) > IoU_threshold:
                # true negative
                TN.append(id)
            else:
                # false negative
                FN.append(id)
    
    tp = len(TP)
    tn = len(TN)
    fp = len(FP)
    fn = len(FN)
    print("total: ", N)
    print("TP, TN, FP, FN: ", tp, tn, fp, fn)
    print("FP: ", FP)
    print("FN: ", FN)
    print("Duplicated ids: ", uniqueness(TP, TN))
    print("accuracy: {acc:.2f}%".format(acc=(tp+tn)/(tp+tn+fp+fn)*100))
    print("precision: {pre:.2f}%".format(pre=tp/(tp+fp)*100))
    print("recall: {rec:.2f}%".format(rec=tp/(tp+fn)*100))
    acc=(tp+tn)/(tp+tn+fp+fn)*100
    pre=tp/(tp+fp)*100
    rec=tp/(tp+fn)*100
    np.save(merged_shapefile[:-4]+"_results.npy", np.array((acc, pre, rec)))


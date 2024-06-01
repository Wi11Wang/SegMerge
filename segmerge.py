import json
import sys
sys.path.append("..")

from numba import njit
import torch
import numpy as np
from tqdm import tqdm


def merge_masks(masks, mode='clamp', device='cuda'):
    """
    merge a list of binary masks in to a single mask
    :param masks: input masks
    :param device: cuda or cpu
    :return: merged mask, number of masks
    """
    assert mode in ['clamp', 'cut']
    n = masks.shape[0]
    mask_values = torch.arange(1, n + 1).view(-1, 1, 1).to(device)
    merged_mask = torch.sum(masks * mask_values, dim=0).to(torch.uint8)
    if mode == 'clamp':
        return torch.clamp(merged_mask, max=n).cpu().numpy()
    elif mode == 'cut':
        merged_mask[merged_mask > n] = 0
        return merged_mask.cpu().numpy()


def _segment(model, image, mode='clamp', device='cuda'):
    model = model.to(device)
    pred = model.predict(image, retina_masks=True, verbose=False, imgsz=1024)
    if pred[0].masks is not None:
        return merge_masks(masks=pred[0].masks.data, mode=mode, device=device)
    else:
        return np.zeros((image.size[1], image.size[0]))


def segment(model, tomo, out, preprocessor=lambda x: x, mode='clamp'):
    n_slices, width, height = tomo.shape
    for i in tqdm(range(n_slices)):
        out[i] = _segment(model, preprocessor(tomo[i]), mode=mode)
        if isinstance(out, np.memmap):
            out.flush()


@njit
def find_possible_matches(mask1, mask2, max_masks_per_img=128, max_label_val=16384, overlap_thresh=0.8):
    idx2label_arr_size = max_masks_per_img + 1
    label2idx_arr_size = max_label_val + 1
    intersection_sizes = np.zeros((idx2label_arr_size, idx2label_arr_size), dtype=np.int32)
    mask1_sizes = np.zeros((idx2label_arr_size,), dtype=np.int32)
    mask2_sizes = np.zeros((idx2label_arr_size,), dtype=np.int32)
    # label to index map
    # i.e. you can you label value to get the index
    mask1_label2idx = np.zeros((label2idx_arr_size,), dtype=np.int16)
    mask2_label2idx = np.zeros((label2idx_arr_size,), dtype=np.int16)
    # index to label map
    # i.e. you can you index to get the label value
    mask1_idx2label = np.zeros((idx2label_arr_size,), dtype=np.int16)
    mask2_idx2label = np.zeros((idx2label_arr_size,), dtype=np.int16)
    # record the unique values in the mask, including the "0"
    mask1_unique_count, mask2_unique_count = int(1), int(1)

    # iterate through mask
    w, h = mask1.shape
    for i in range(w):
        for j in range(h):
            mask1_val, mask2_val = mask1[i, j], mask2[i, j]
            # this is the first time we met "mask1_val", create mapping relationship
            if mask1_val > 0 and mask1_label2idx[mask1_val] == 0:
                mask1_label2idx[mask1_val] = mask1_unique_count
                mask1_idx2label[mask1_unique_count] = mask1_val
                mask1_unique_count += 1
            if mask2_val > 0 and mask2_label2idx[mask2_val] == 0:
                mask2_label2idx[mask2_val] = mask2_unique_count
                mask2_idx2label[mask2_unique_count] = mask2_val
                mask2_unique_count += 1
                # update the size of the corresponding area
            intersection_sizes[mask1_label2idx[mask1_val], mask2_label2idx[mask2_val]] += 1
            mask1_sizes[mask1_label2idx[mask1_val]] += 1
            mask2_sizes[mask2_label2idx[mask2_val]] += 1

    # find the possible match based on dice score and overlap
    possible_matches = []
    for i in range(1, mask1_unique_count):
        curr_matches = []
        for j in range(1, mask2_unique_count):
            # check the overlap rate
            if mask2_sizes[j] > 0 and mask1_sizes[i] + mask2_sizes[j] > 0:
                overlap = intersection_sizes[i, j] / mask2_sizes[j]
                if overlap > overlap_thresh:
                    dice = 2 * intersection_sizes[i, j] / (mask1_sizes[i] + mask2_sizes[j])
                    curr_matches.append([mask2_idx2label[j], overlap, dice])
        possible_matches.append(curr_matches)
    return (possible_matches,
            mask1_idx2label[1:mask1_unique_count],
            mask2_idx2label[:mask2_unique_count],
            mask2_label2idx)


def _dice(mask1, mask2, mask1_label, mask2_val_mask, device):
    m1 = torch.from_numpy(mask1.copy()).to(device)
    m2 = torch.from_numpy(mask2.copy()).to(device)
    mask2_val_mask = torch.from_numpy(mask2_val_mask).to(device)
    m1 = m1 == mask1_label
    m2 = torch.isin(m2, mask2_val_mask, assume_unique=True)
    return 2 * torch.sum(torch.logical_and(m1, m2)) / (torch.sum(m1) + torch.sum(m2))


def update_label_pair(mask2_matches, mask1_idx2_label, mask2_idx2label, mask2_label2idx, mask1, mask2,
                      max_unmatched_label, indices, snn=None, overlap_thresh=0.9, dice_thresh=0.85, top_to_bottom=True):
    """
    1. iterate through mask1 labels
    2.   iterate through mask2 possible matches
    3.      - 0 match -> skip
            - 1 match -> check if above threshold
            - 2+ matches -> find the best match
    """
    # dice based method or siamese neural network
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # matched value
    mask2_n_masks = len(mask2_idx2label) - 1
    matches = np.zeros((mask2_n_masks + 1,), dtype=np.int16)
    idx_dice_map = np.zeros((mask2_n_masks + 1,), dtype=np.float32)
    unmatched_labels = set(mask2_idx2label)
    unmatched_labels.discard(0)
    # number of particles after updating label pair
    # e.g. if there are two masks being merged into one: mask2_cnt - 1 = updated_cnt
    for i, mask1_label in enumerate(mask1_idx2_label):
        possible_matches = mask2_matches[i]
        n_possible_matches = len(possible_matches)
        # 0 possible match -> skip
        if n_possible_matches == 0:
            continue
        # 1 possible match -> mark match if above threshold
        if n_possible_matches == 1 and top_to_bottom:
            possible_matched_label = int(possible_matches[0][0])
            possible_matched_label_idx = mask2_label2idx[possible_matched_label]
            dice = possible_matches[0][1]
            # check if above threshold and if the current label is already matched
            if possible_matches[0][1] > overlap_thresh and dice > dice_thresh and dice > idx_dice_map[possible_matched_label_idx]:
                matches[possible_matched_label_idx] = mask1_label
                idx_dice_map[possible_matched_label_idx] = dice
                max_unmatched_label = max(mask1_label, max_unmatched_label - 1) + 1
                unmatched_labels.discard(possible_matched_label)
        # 2 possible matches -> find the best match
        if n_possible_matches >= 2:
            possible_matches_arr = np.array(mask2_matches[i])
            best_match = [possible_matches_arr[np.argmax(possible_matches_arr[:, 2]), 0]]
            best_dice = np.max(possible_matches_arr[:, 2])
            possible_match_labels = possible_matches_arr[:, 0].reshape(-1)
            # iterate through combinations to find the best match
            for indices in indices[n_possible_matches - 2]:
                mask2_selected_labels = possible_match_labels[indices]
                dice = _dice(mask1, mask2, mask1_label, mask2_selected_labels, device).item()
                if dice > best_dice:
                    best_dice = dice
                    best_match = mask2_selected_labels
            if best_dice > dice_thresh:
                max_unmatched_label = max(mask1_label, max_unmatched_label - 1) + 1
                for possible_matched_label in best_match:
                    possible_matched_label_idx = mask2_label2idx[int(possible_matched_label)]
                    if best_dice > idx_dice_map[possible_matched_label_idx]:
                        matches[possible_matched_label_idx] = mask1_label
                        idx_dice_map[possible_matched_label_idx] = best_dice
                        unmatched_labels.discard(possible_matched_label)
    if top_to_bottom:
        # Set labels for unmatched masks
        for unmatched_label in unmatched_labels:
            matches[mask2_label2idx[unmatched_label]] = max_unmatched_label
            max_unmatched_label += 1
    return matches, max_unmatched_label


@njit
def update_labels(mask2, mask2_idx2matches, mask2_label2idx):
    merged = mask2.copy()
    w, h = mask2.shape
    for i in range(w):
        for j in range(h):
            merged[i, j] = mask2_idx2matches[mask2_label2idx[mask2[i, j]]]
    return merged


def merge(mask, out=None, top_to_bottom=True, max_particles_per_img=128, max_label_val=16384, overlap_thresh=0.9, dice_thresh=0.85):
    # load combination of indices
    with open('indices.json', mode='r') as f:
        indices = json.load(f)

    n_slices = mask.shape[0]
    max_unmatched_label = 1

    if top_to_bottom:
        merge_range = range(1, n_slices)
        top_slice = mask[0]
    else:
        merge_range = range(n_slices - 2, -1, -1)
        top_slice = mask[n_slices - 1]

    if out is None:
        out_arr = mask
    else:
        out_arr = out
    out_type = 0
    if isinstance(out_arr, np.memmap):
        out_type = 1

    for slice_idx in tqdm(merge_range):
        bot_slice = mask[slice_idx]
        (bot_slice_possible_matches,
         top_slice_idx2label,
         bot_slice_idx2label,
         bot_slice_label2idx) = find_possible_matches(mask1=top_slice, mask2=bot_slice,
                                                      max_masks_per_img=max_particles_per_img,
                                                      max_label_val=max_label_val,
                                                      overlap_thresh=overlap_thresh)
        (bot_slice_matches,
         max_unmatched_label) = update_label_pair(mask2_matches=bot_slice_possible_matches,
                                                  mask1_idx2_label=top_slice_idx2label,
                                                  mask2_idx2label=bot_slice_idx2label,
                                                  mask2_label2idx=bot_slice_label2idx, mask1=top_slice,
                                                  mask2=bot_slice,
                                                  overlap_thresh=overlap_thresh,
                                                  dice_thresh=dice_thresh,
                                                  max_unmatched_label=max_unmatched_label,
                                                  indices=indices)
        top_slice = update_labels(mask2=bot_slice,
                                  mask2_idx2matches=bot_slice_matches,
                                  mask2_label2idx=bot_slice_label2idx)
        out_arr[slice_idx] = top_slice
        if out_type == 1:
            out_arr.flush()


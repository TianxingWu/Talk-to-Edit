import argparse
import copy
import logging
import os

import numpy as np
import torch

from models import create_model
from models.utils import save_image
from utils.editing_utils import edit_target_attribute
from utils.inversion_utils import inversion
from utils.logger import get_root_logger
from utils.options import (dict2str, dict_to_nonedict, parse,
                           parse_opt_wrt_resolution)
from utils.util import make_exp_dirs

from tqdm import tqdm
import csv


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    # parser.add_argument('--attr', type=str, help='Attribute to be edited.')
    # parser.add_argument('--target_val', type=int, help='Target Attribute Value.')
    parser.add_argument('--attrs', nargs='+', type=str, help='Attributes to be edited.')
    parser.add_argument('--target_vals', nargs='+', type=int, default=None, help='Target Attribute Values.')
    parser.add_argument('--target_val_changes', nargs='+', type=int, default=None,
                        help='Target Attribute Value Changes.')

    return parser.parse_args()


def edit_one_seq(opt, idx_to_attr, editing_logger, field_model,
                 edit_idx_seq, target_scores, attribute_dict, pred_score, latent_code, edited_latent_code):
    for idx in edit_idx_seq:
        edit_label = {'attribute': idx_to_attr[idx], 'target_score': target_scores[idx],
                      'target_score_change': None}
        round_idx = 0
        attribute_dict, pred_score, exception_mode, latent_code, edited_latent_code, saved_image = edit_target_attribute(
            opt, attribute_dict, pred_score, edit_label, round_idx, latent_code,
            edited_latent_code, field_model, editing_logger,
            print_intermediate_result=False)
        # ---------- skip results with exception ----------
        if exception_mode != 'normal':
            if exception_mode == 'already_at_target_class':
                # editing_logger.info(f"already_at_target_class. <Skip>")
                break
            elif exception_mode == 'max_edit_num_reached':
                # editing_logger.info(f"max_edit_num_reached. <Skip>")
                break
            elif exception_mode == 'current_class_not_clear':
                # editing_logger.info(f"current_class_not_clear. <Skip>")
                break
            elif exception_mode == 'confidence_low':
                # editing_logger.info(f"confidence_low. <Skip>")
                break
    return attribute_dict, pred_score, exception_mode, latent_code, edited_latent_code, saved_image


def set_target_scores(target_scores, attribute_dict, edit_idx_set):
    if 0 in edit_idx_set:
        if attribute_dict['Bangs'] == 0:
            # target_scores[0] = np.random.choice([1, 2, 3, 4, 5])
            # target_scores[0] = np.random.choice([1, 2, 3])
            target_scores[0] = np.random.choice([1, 2])

        else:
            target_scores[0] = 0
    if 1 in edit_idx_set:
        if attribute_dict['Eyeglasses'] == 0:
            # target_scores[1] = np.random.choice([1, 2, 3])
            target_scores[1] = np.random.choice([1, 2])
        else:
            target_scores[1] = 0
    if 2 in edit_idx_set:
        if attribute_dict['No_Beard'] == 0:
            # target_scores[2] = np.random.choice([1, 2, 3])
            target_scores[2] = np.random.choice([1, 2])
        else:
            target_scores[2] = 0
    if 3 in edit_idx_set:
        # if attribute_dict['Smiling'] == 0:
        if attribute_dict['Smiling'] <= 1:
            # target_scores[3] = np.random.choice([1, 2, 3])
            target_scores[3] = np.random.choice([2, 3])
        else:
            # if attribute_dict['Smiling'] >= 3:
            #     target_scores[3] = 1
            # else:
            #     target_scores[3] = np.random.choice([0, 1])
            target_scores[3] = 1
    if 4 in edit_idx_set:
        if attribute_dict['Young'] <= 2:
            # target_scores[4] = np.random.choice([3, 4, 5])
            target_scores[4] = np.random.choice([3, 4])
        else:
            # target_scores[4] = np.random.choice([0, 1, 2])
            target_scores[4] = np.random.choice([1, 2])

    return target_scores


def main():
    # ---------- Set up -----------
    args = parse_args()
    np.random.seed(10032)

    opt = parse(args.opt, is_train=False)
    opt = parse_opt_wrt_resolution(opt)
    # args = parse_args_from_opt(args, opt)
    make_exp_dirs(opt)

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # set up logger
    save_log_path = f'{opt["path"]["log"]}'
    editing_logger = get_root_logger(
        logger_name='editing',
        log_level=logging.INFO,
        log_file=f'{save_log_path}/editing.log')
    editing_logger.info(dict2str(opt))

    # ---------- create model ----------
    field_model = create_model(opt)

    # ---------- load latent code ----------
    random_codes = np.load(opt['latent_code_path'])
    # os.makedirs(os.path.join(opt["path"]["results_root"], 'orig'), exist_ok=True)

    # ---------- for different editing modes ----------
    target_scores = None
    target_score_changes = None

    if args.target_vals and not args.target_val_changes:
        target_scores = args.target_vals
    elif args.target_val_changes and not args.target_vals:
        target_score_changes = args.target_val_changes
    else:
        editing_logger.info("No input target_vals or target_val_changes, start binary editing.")

    # ---------- load editing settings ----------
    # edit_idx_seqs = np.array(opt['edit_idx_seqs'])
    idx_to_attr = opt['idx_to_attr']  # for print info
    # idx_to_attr = ['Bangs', 'Eyeglasses', 'No_Beard', 'Smiling', 'Young']
    print_intermediate_result = False

    # make image save paths
    save_image_paths_dict = {'orig': os.path.join(opt["path"]["results_root"], 'orig'),
                             'edited': os.path.join(opt["path"]["results_root"], 'edited'),
                             'recover_good': os.path.join(opt["path"]["results_root"], 'recover_good'),
                             'recover_bad': os.path.join(opt["path"]["results_root"], 'recover_bad')}
    for path in save_image_paths_dict.values():
        os.makedirs(path, exist_ok=True)
    # make results csv
    csv_path = os.path.join(opt["path"]["results_root"], 'results.csv')
    with open(csv_path, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'code_idx', 'inv_seq_good', 'inv_seq_bad',
                                               'label_orig', 'label_edited', 'label_recover_good', 'label_recover_bad'])
        writer.writeheader()

    # make latent_code list
    orig_latent_codes = []
    edited_latent_codes = []
    recover_latent_good_codes = []
    recover_latent_bad_codes = []
    orig_edited_latent_codes = []
    edited_edited_llatent_codes = []
    recover_edited_llatent_good_codes = []
    recover_edited_llatent_bad_codes = []


    # ---------- initialize data dictionaries ----------
    # attribute_dict = {
    #     "Bangs": 0,
    #     "Eyeglasses": 0,
    #     "No_Beard": 0,
    #     "Smiling": 0,
    #     "Young": 0
    # }

    orig_data = {'latent_code': None,
                 'edited_latent_code': None,
                 'saved_img': None,
                 'attribute_dict': None,
                 'label': None,
                 'pred_score': None
                 }
    edited_data = copy.deepcopy(orig_data)
    recover_good_data = copy.deepcopy(orig_data)
    recover_bad_data = copy.deepcopy(orig_data)

    # -------------------------------------
    # ---------- MAIN EDIT LOOP -----------
    # -------------------------------------
    success_cnt = 0
    pbar = tqdm(range(len(random_codes)))
    for code_idx in pbar:
        pbar.set_postfix({'success_cnt': success_cnt})
        if success_cnt == 100:
            break
        random_code = random_codes[code_idx]
        # ---------- get original latent_code and labels ----------
        random_code = torch.from_numpy(random_code).to(torch.device('cuda'))
        with torch.no_grad():
            latent_code = field_model.stylegan_gen.get_latent(random_code)
        latent_code = latent_code.cpu().numpy()
        # orig_code = copy.deepcopy(latent_code)
        orig_data['latent_code'] = latent_code
        # ---------- synthesize images and predict label ----------
        with torch.no_grad():
            orig_img, orig_label, orig_score = \
                field_model.synthesize_and_predict(torch.from_numpy(latent_code).to(torch.device('cuda')))
        # select confident images
        if np.any(np.array(orig_score) < opt['confidence_thresh_input']):
            # editing_logger.info(f"Not confident about original attribute class. <SKIP>")
            editing_logger.info(f"\ncode {code_idx}: BAD INPUT 1")
            continue
        orig_data['saved_img'] = orig_img
        orig_data['label'] = orig_label
        orig_data['pred_score'] = orig_score
        # ---------- read labels ----------
        orig_data['attribute_dict'] = {
            "Bangs": 0,
            "Eyeglasses": 0,
            "No_Beard": 0,
            "Smiling": 0,
            "Young": 0
        }
        orig_data['attribute_dict']['Bangs'] = orig_label[0]
        orig_data['attribute_dict']['Eyeglasses'] = orig_label[1]
        orig_data['attribute_dict']['No_Beard'] = orig_label[2]
        orig_data['attribute_dict']['Smiling'] = orig_label[3]
        orig_data['attribute_dict']['Young'] = orig_label[4]
        # ---------- restrictions ----------
        if orig_data['attribute_dict']['Bangs'] > 5 or \
                orig_data['attribute_dict']['Eyeglasses'] > 3 or \
                orig_data['attribute_dict']['No_Beard'] > 3 or \
                orig_data['attribute_dict']['Smiling'] > 4 or \
                orig_data['attribute_dict']['Young'] > 5:
            # editing_logger.info(f"Restricted original attribute class. <Skip sample #{code_idx}>")
            editing_logger.info(f"\ncode {code_idx}: BAD INPUT 2")
            continue

        # ---------- generate editing sequence ----------
        forward_seq = np.random.choice([0, 1, 2, 3, 4], size=3, replace=False)
        inv_seq_good = np.flip(forward_seq)
        inv_seq_bad = np.random.choice(inv_seq_good, size=3, replace=False)
        while np.all(inv_seq_bad == inv_seq_good):
            inv_seq_bad = np.random.choice(inv_seq_good, size=3, replace=False)

        # ---------- forward editing  ----------
        target_scores = [None, None, None, None, None]
        target_scores = set_target_scores(target_scores, orig_data['attribute_dict'], set(forward_seq))
        attribute_dict, pred_score, exception_mode, latent_code, edited_latent_code, saved_image = \
            edit_one_seq(opt, idx_to_attr, editing_logger, field_model,
                         edit_idx_seq=forward_seq,
                         target_scores=target_scores,
                         attribute_dict=orig_data['attribute_dict'],
                         pred_score=orig_data['pred_score'],
                         latent_code=orig_data['latent_code'],
                         edited_latent_code=None)
        if exception_mode == 'normal':  # OR already_at_target_class?
            edited_data['latent_code'] = latent_code
            edited_data['edited_latent_code'] = edited_latent_code
            edited_data['saved_img'] = saved_image
            edited_data['attribute_dict'] = attribute_dict
            edited_data['label'] = list(attribute_dict.values())
            edited_data['pred_score'] = pred_score
        else:
            editing_logger.info(f"\ncode {code_idx}: BAD EDIT")
            continue

        # ---------- backward editing settings ----------
        target_scores = [None, None, None, None, None]
        # target is set based on original label and edited parts
        for idx in inv_seq_good:
            target_scores[idx] = orig_label[idx]

        # ---------- backward editing (good) ----------
        attribute_dict, pred_score, exception_mode, latent_code, edited_latent_code, saved_image = \
            edit_one_seq(opt, idx_to_attr, editing_logger, field_model,
                         edit_idx_seq=inv_seq_good,
                         target_scores=target_scores,
                         attribute_dict=edited_data['attribute_dict'],
                         pred_score=edited_data['pred_score'],
                         latent_code=edited_data['latent_code'],
                         edited_latent_code=edited_data['edited_latent_code'])
        if exception_mode == 'normal':
            recover_good_data['latent_code'] = latent_code
            recover_good_data['edited_latent_code'] = edited_latent_code
            recover_good_data['saved_img'] = saved_image
            recover_good_data['attribute_dict'] = attribute_dict
            recover_good_data['label'] = list(attribute_dict.values())
            recover_good_data['pred_score'] = pred_score
        else:
            editing_logger.info(f"\ncode {code_idx}: BAD RECOVER 1")
            continue

        # ---------- backward editing (bad) ----------
        attribute_dict, pred_score, exception_mode, latent_code, edited_latent_code, saved_image = \
            edit_one_seq(opt, idx_to_attr, editing_logger, field_model,
                         edit_idx_seq=inv_seq_bad,
                         target_scores=target_scores,
                         attribute_dict=edited_data['attribute_dict'],
                         pred_score=edited_data['pred_score'],
                         latent_code=edited_data['latent_code'],
                         edited_latent_code=edited_data['edited_latent_code'])
        if exception_mode == 'normal':
            recover_bad_data['latent_code'] = latent_code
            recover_bad_data['edited_latent_code'] = edited_latent_code
            recover_bad_data['saved_img'] = saved_image
            recover_bad_data['attribute_dict'] = attribute_dict
            recover_bad_data['label'] = list(attribute_dict.values())
            recover_bad_data['pred_score'] = pred_score
        else:
            editing_logger.info(f"\ncode {code_idx}: BAD RECOVER 2")
            continue

        # ---------- save success data ----------
        success_cnt += 1
        # save img
        save_image(orig_data['saved_img'], os.path.join(save_image_paths_dict['orig'], f'{success_cnt:03d}.jpg'))
        save_image(edited_data['saved_img'], os.path.join(save_image_paths_dict['edited'], f'{success_cnt:03d}.jpg'))
        save_image(recover_good_data['saved_img'], os.path.join(save_image_paths_dict['recover_good'], f'{success_cnt:03d}.jpg'))
        save_image(recover_bad_data['saved_img'], os.path.join(save_image_paths_dict['recover_bad'], f'{success_cnt:03d}.jpg'))
        # save latent code
        orig_latent_codes.append(orig_data['latent_code'])
        edited_latent_codes.append(edited_data['latent_code'])
        recover_latent_good_codes.append(recover_good_data['latent_code'])
        recover_latent_bad_codes.append(recover_bad_data['latent_code'])
        orig_edited_latent_codes.append(orig_data['edited_latent_code'])
        edited_edited_llatent_codes.append(edited_data['edited_latent_code'])
        recover_edited_llatent_good_codes.append(recover_good_data['edited_latent_code'])
        recover_edited_llatent_bad_codes.append(recover_bad_data['edited_latent_code'])
        # write csv
        with open(csv_path, mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'code_idx', 'inv_seq_good', 'inv_seq_bad',
                           'label_orig', 'label_edited', 'label_recover_good', 'label_recover_bad'])
            writer.writerow({'filename': f'{success_cnt:03d}.jpg',
                             'code_idx': code_idx,
                             'inv_seq_good': inv_seq_good,
                             'inv_seq_bad': inv_seq_bad,
                             'label_orig': orig_data['label'],
                             'label_edited': edited_data['label'],
                             'label_recover_good': recover_good_data['label'],
                             'label_recover_bad': recover_bad_data['label']})

    np.save('orig_latent_codes', orig_latent_codes)
    np.save('edited_latent_codes', edited_latent_codes)
    np.save('recover_latent_good_codes', recover_latent_good_codes)
    np.save('recover_latent_bad_codes', recover_latent_bad_codes)

    np.save('orig_edited_latent_codes', orig_edited_latent_codes)
    np.save('edited_edited_llatent_codes', edited_edited_llatent_codes)
    np.save('recover_edited_llatent_good_codes', recover_edited_llatent_good_codes)
    np.save('recover_edited_llatent_bad_codes', recover_edited_llatent_bad_codes)

if __name__ == '__main__':
    main()

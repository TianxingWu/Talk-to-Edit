import argparse
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


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    # parser.add_argument('--attr', type=str, help='Attribute to be edited.')
    # parser.add_argument('--target_val', type=int, help='Target Attribute Value.')
    parser.add_argument('--attrs', nargs='+', type=str, help='Attributes to be edited.')
    parser.add_argument('--target_vals', nargs='+', type=int, default=None, help='Target Attribute Values.')
    parser.add_argument('--target_val_changes', nargs='+', type=int, default=None, help='Target Attribute Value Changes.')

    return parser.parse_args()


def set_target_scores(target_scores, attribute_dict, edit_idx_set):
    if 0 in edit_idx_set:
        if attribute_dict['Bangs'] == 0:
            # target_scores[0] = np.random.choice([1, 2, 3, 4, 5])
            target_scores[0] = np.random.choice([1, 2, 3])
        else:
            target_scores[0] = 0
    if 1 in edit_idx_set:
        if attribute_dict['Eyeglasses'] == 0:
            target_scores[1] = np.random.choice([1, 2, 3])
            # target_scores[1] = np.random.choice([1, 2])
        else:
            target_scores[1] = 0
    if 2 in edit_idx_set:
        if attribute_dict['No_Beard'] == 0:
            target_scores[2] = np.random.choice([1, 2, 3])
            # target_scores[2] = np.random.choice([1, 2])
        else:
            target_scores[2] = 0
    if 3 in edit_idx_set:
        # if attribute_dict['Smiling'] == 0:
        if attribute_dict['Smiling'] <= 1:
            # target_scores[3] = np.random.choice([1, 2, 3])
            target_scores[3] = np.random.choice([2, 3])
        else:
            if attribute_dict['Smiling'] >= 3:
                target_scores[3] = 1
            else:
                target_scores[3] = np.random.choice([0, 1])
            # target_scores[3] = 0
            # target_scores[3] = 1
            # target_scores[3] = np.random.choice([0, 1])
    if 4 in edit_idx_set:
        if attribute_dict['Young'] <= 2:
            target_scores[4] = np.random.choice([3, 4, 5])
        else:
            target_scores[4] = np.random.choice([0, 1, 2])

    return target_scores


def edit_loop(opt,
              random_codes,
              edit_idx_seqs,
              num_edit_type,
              field_model,
              base_latent_codes,
              base_edited_latent_codes,
              base_labels,
              base_scores,
              base_idxes,
              idx_to_attr,
              editing_logger,
              save_image_paths,
              max_per_edit):
    edit_idx_seq = edit_idx_seqs[num_edit_type]
    if len(edit_idx_seq) == 1:  # base edit
        is_base = True
    else:
        edit_idx_seq = edit_idx_seq[1:]
        is_base = False

    # ---------- initialize attribute_dict (labels) ----------
    attribute_dict = {
        "Bangs": 0,
        "Eyeglasses": 0,
        "No_Beard": 0,
        "Smiling": 0,
        "Young": 0
    }

    NUM_SOURCE = len(random_codes) if is_base else len(base_idxes)
    cnt = 0
    pbar = tqdm(range(NUM_SOURCE))
    for i in pbar:
        pbar.set_postfix({'cnt_generated': cnt})
        # if already reach max num, stop editing

        # # ========================================================= DEBUG ONLY ====================================
        # if is_base and cnt == 100:
        #     break
        # # ==========================================================================================================

        if (not is_base) and cnt == max_per_edit:
            break
        code_idx = i if is_base else base_idxes[i]
        # editing_logger.info(f"================================================================== #{code_idx}")
        # ---------- get current latent_code and labels ----------
        if is_base:
            random_code = random_codes[code_idx, :]
            random_code = torch.from_numpy(random_code).to(torch.device('cuda'))
            with torch.no_grad():
                latent_code = field_model.stylegan_gen.get_latent(random_code)
            latent_code = latent_code.cpu().numpy()
            edited_latent_code = None
            # ---------- synthesize images and predict label ----------
            with torch.no_grad():
                _, start_label, start_score = \
                    field_model.synthesize_and_predict(torch.from_numpy(latent_code).to(torch.device('cuda')))
            # select confident images (only in base edit)
            if np.any(np.array(start_score) < opt['confidence_thresh_input']):
                # editing_logger.info(f"Not confident about original attribute class. <SKIP>")
                continue
        else:
            latent_code = base_latent_codes[i]
            edited_latent_code = base_edited_latent_codes[i]
            start_label = base_labels[i]
            start_score = base_scores[i]

        # ---------- read labels ----------
        attribute_dict['Bangs'] = start_label[0]
        attribute_dict['Eyeglasses'] = start_label[1]
        attribute_dict['No_Beard'] = start_label[2]
        attribute_dict['Smiling'] = start_label[3]
        attribute_dict['Young'] = start_label[4]

        # ---------- restrictions (only in base edit) ----------
        if is_base:
            if attribute_dict['Bangs'] > 5 or \
                    attribute_dict['Eyeglasses'] > 3 or \
                    attribute_dict['No_Beard'] > 3 or \
                    attribute_dict['Smiling'] > 4 or \
                    attribute_dict['Young'] > 5:
                # editing_logger.info(f"Restricted original attribute class. <Skip sample #{code_idx}>")
                continue

        # ---------- set edit target value ----------
        target_scores = [None, None, None, None, None]
        target_scores = set_target_scores(target_scores, attribute_dict, set(edit_idx_seq))

        # ---------- start to edit ----------
        # editing_logger.info(f'\nstart_label: {start_label}, start_score: {start_score}')

        # initialize predicted labels and scores
        attribute_dict = attribute_dict  # attribute class will change during sequential editing
        pred_score = start_score

        for idx in edit_idx_seq:
            edit_label = {'attribute': idx_to_attr[idx], 'target_score': target_scores[idx], 'target_score_change': None}

            round_idx = 0
            # editing_logger.info(f'\ntarget attribute: {idx_to_attr[idx]} ({idx})')
            # editing_logger.info(f'current cls: {attribute_dict[idx_to_attr[idx]]}')
            # editing_logger.info(f'target cls: {target_scores[idx]}')

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

        # ---------- save final edited image and update variables if it's a success edit ----------
        if exception_mode == 'normal':
            save_image(saved_image, f'{save_image_paths[num_edit_type]}/{code_idx:06d}.jpg')
            if is_base:
                base_latent_codes.append(latent_code)
                base_edited_latent_codes.append(edited_latent_code)
                base_labels.append(list(attribute_dict.values()))
                base_scores.append(pred_score)
                base_idxes.append(code_idx)
            cnt += 1


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

    # save_image_path = f'{opt["path"]["visualization"]}'
    # os.makedirs(save_image_path, exist_ok=True)

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
    edit_idx_seqs = np.array(opt['edit_idx_seqs'])
    idx_to_attr = opt['idx_to_attr']  # for print info
    # idx_to_attr = ['Bangs', 'Eyeglasses', 'No_Beard', 'Smiling', 'Young']
    print_intermediate_result = False

    # make image save paths
    save_image_paths = []
    for edit_idx_seq in edit_idx_seqs:
        str_edit_types = []
        for i in edit_idx_seq:
            if idx_to_attr[i] == 'No_Beard':
                str_edit_types.append('Beard')
            else:
                str_edit_types.append(idx_to_attr[i])
        str_edit_types = '-'.join(str_edit_types)
        save_image_path = os.path.join(opt["path"]["results_root"], f'{str_edit_types}')
        os.makedirs(save_image_path, exist_ok=True)
        save_image_paths.append(save_image_path)

    # -------------------------------------
    # ---------- MAIN EDIT LOOP -----------
    # -------------------------------------

    for num_edit_type in range(len(edit_idx_seqs)):
        editing_logger.info(f"Start Edit Sequence: {edit_idx_seqs[num_edit_type]}")
        if num_edit_type == 0:  # base edit
            base_latent_codes = []
            base_edited_latent_codes = []
            base_labels = []
            base_scores = []
            base_idxes = []
        else:
            # random choose from base-edited images
            temp = np.random.permutation(len(base_idxes))
            base_latent_codes = [base_latent_codes[x] for x in temp]
            base_edited_latent_codes = [base_edited_latent_codes[x] for x in temp]
            base_labels = [base_labels[x] for x in temp]
            base_scores = [base_scores[x] for x in temp]
            base_idxes = [base_idxes[x] for x in temp]

        edit_loop(opt,
                  random_codes,
                  edit_idx_seqs,
                  num_edit_type,
                  field_model,
                  base_latent_codes,
                  base_edited_latent_codes,
                  base_labels,
                  base_scores,
                  base_idxes,
                  idx_to_attr,
                  editing_logger,
                  save_image_paths,
                  max_per_edit=5000)
    #
    #
    #
    #
    # for code_idx in tqdm(range(len(random_codes))):
    #     editing_logger.info(f"================================================================== #{code_idx}")
    #     # ---------- get orig latent_code ----------
    #     random_code = random_codes[code_idx, :]
    #     random_code = torch.from_numpy(random_code).to(torch.device('cuda'))
    #     with torch.no_grad():
    #         latent_code_orig = field_model.stylegan_gen.get_latent(random_code)
    #     latent_code_orig = latent_code_orig.cpu().numpy()
    #
    #     # ---------- synthesize images and predict label ----------
    #     with torch.no_grad():
    #         start_image, start_label, start_score = \
    #             field_model.synthesize_and_predict(torch.from_numpy(latent_code_orig).to(torch.device('cuda')))
    #     # select confident images (only in base edit)
    #     if np.any(np.array(start_score) < opt['confidence_thresh_input']):
    #         editing_logger.info(f"Not confident about original attribute class. <SKIP>")
    #         continue
    #     # initialize attribute_dict
    #     attribute_dict = {
    #         "Bangs": start_label[0],
    #         "Eyeglasses": start_label[1],
    #         "No_Beard": start_label[2],
    #         "Smiling": start_label[3],
    #         "Young": start_label[4],
    #     }
    #     # restrictions (only in base edit)
    #     if attribute_dict['Bangs'] > 5 or \
    #         attribute_dict['Eyeglasses'] > 3 or \
    #         attribute_dict['No_Beard'] > 3 or \
    #         attribute_dict['Smiling'] > 4 or \
    #         attribute_dict['Young'] > 5:
    #         editing_logger.info(f"Restricted original attribute class. <Skip sample #{code_idx}>")
    #         continue
    #
    #     # ---------- set edit target ----------
    #     if target_scores is None:
    #         target_scores = [None, None, None, None, None]
    #         target_scores = set_target_scores(target_scores, attribute_dict, {base_edit_idx})
    #
    #     # ---------- operate base edit ----------
    #     latent_code = latent_code_orig
    #     edited_latent_code = None
    #     # attribute_dict_temp = attribute_dict  # ????
    #     # pred_score_temp = start_score   # ?????
    #     editing_logger.info(f'\nstart_label: {start_label}, start_score: {start_score}')
    #
    #     i = base_edit_idx
    #     edit_label = {'attribute': idx_to_attr[i], 'target_score': target_scores[i], 'target_score_change': None}
    #
    #     round_idx = 0
    #     editing_logger.info(f'current cls: {attribute_dict_temp[idx_to_attr[i]]}')
    #     editing_logger.info(f'target cls: {target_scores[i]}')
    #
    #     attribute_dict_temp, pred_score_temp, exception_mode, latent_code, edited_latent_code, saved_image = edit_target_attribute(
    #         opt, attribute_dict, start_score, edit_label, round_idx, latent_code,
    #         edited_latent_code, field_model, editing_logger,
    #         print_intermediate_result)
    #
    #     # ---------- save results with no exception ----------
    #     if exception_mode != 'normal':
    #         if exception_mode == 'already_at_target_class':
    #             editing_logger.info(f"already_at_target_class. <Skip>")
    #             break  ############################
    #         elif exception_mode == 'max_edit_num_reached':
    #             editing_logger.info(f"max_edit_num_reached. <Skip>")
    #             break
    #         elif exception_mode == 'current_class_not_clear':
    #             editing_logger.info(f"current_class_not_clear. <Skip>")
    #             break
    #         elif exception_mode == 'confidence_low':
    #             editing_logger.info(f"confidence_low. <Skip>")
    #             break
    #
    #     # str_edit_types = '-'.join([idx_to_attr[k] for k in edit_idx_seq[:j+1]])
    #     str_edit_types = []
    #     for k in edit_idx_seq[:j + 1]:
    #         if idx_to_attr[k] == 'No_Beard':
    #             str_edit_types.append('Beard')
    #         else:
    #             str_edit_types.append(idx_to_attr[k])
    #     str_edit_types = '-'.join(str_edit_types)
    #     save_image_path = os.path.join(opt["path"]["results_root"], f'{str_edit_types}')
    #     os.makedirs(save_image_path, exist_ok=True)
    #     if saved_image is not None:
    #         save_image(saved_image, f'{save_image_path}/{code_idx:06d}.png')
    #
    # if exception_mode == 'normal':
    #     good_list.append(code_idx)
    #
    # editing_logger.info('\n\n=============================')
    # editing_logger.info('finial good_list:')
    # editing_logger.info(good_list)
    #
    #
    #
    #
    #
    #
    #
    #
    # good_list = []
    # for code_idx in tqdm(range(len(random_codes))):
    #     editing_logger.info(f"================================================================== #{code_idx}")
    #     random_code = random_codes[code_idx, :]
    #     random_code = torch.from_numpy(random_code).to(torch.device('cuda'))
    #     with torch.no_grad():
    #         latent_code_orig = field_model.stylegan_gen.get_latent(random_code)
    #     latent_code_orig = latent_code_orig.cpu().numpy()
    #
    #     # ---------- synthesize images ----------
    #     with torch.no_grad():
    #         start_image, start_label, start_score = \
    #             field_model.synthesize_and_predict(torch.from_numpy(latent_code_orig).to(torch.device('cuda')))  # noqa
    #
    #     # select confident images
    #     # if np.any(np.array(start_score) < opt['confidence_thresh']):
    #     if np.any(np.array(start_score) < opt['confidence_thresh_input']):
    #         editing_logger.info(f"Not confident about original attribute class. <SKIP #{code_idx}>")
    #         continue
    #
    #     # initialize attribute_dict
    #     attribute_dict = {
    #         "Bangs": start_label[0],
    #         "Eyeglasses": start_label[1],
    #         "No_Beard": start_label[2],
    #         "Smiling": start_label[3],
    #         "Young": start_label[4],
    #     }
    #
    #     # restrictions ????????????????? or in base_model.py ?
    #     if attribute_dict['Bangs'] > 5 or \
    #         attribute_dict['Eyeglasses'] > 3 or \
    #         attribute_dict['No_Beard'] > 3 or \
    #         attribute_dict['Smiling'] > 4 or \
    #         attribute_dict['Young'] > 5:
    #         editing_logger.info(f"Restricted original attribute class. <Skip sample #{code_idx}>")
    #         continue
    #
    #     save_image(start_image, os.path.join(opt["path"]["results_root"], 'orig', f'{code_idx:06d}.png'))
    #
    #     if target_scores is None:
    #         target_scores = [None, None, None, None, None]
    #         target_scores = set_target_scores(target_scores, attribute_dict, set(edit_idx_seq))
    #
    #
    #
    #
    #     for edit_idx_seq in edit_idx_seqs:
    #         latent_code = latent_code_orig.copy()
    #         edited_latent_code = None
    #         # import pdb
    #         # pdb.set_trace()
    #         attribute_dict_temp = copy.deepcopy(attribute_dict)  # attribute class changing
    #         pred_score_temp = copy.deepcopy(pred_score)
    #         editing_logger.info(f'\nstart_label: {start_label}, start_score: {start_score}')
    #         for j in range(len(edit_idx_seq)):
    #             i = edit_idx_seq[j]
    #             if target_scores:  # absolute target
    #                 edit_label = {'attribute': idx_to_attr[i], 'target_score': target_scores[i], 'target_score_change': None}
    #             elif target_score_changes:  # relative target
    #                 edit_label = {'attribute': idx_to_attr[i], 'target_score': None, 'target_score_change': target_score_changes[i]}
    #             else:  # binary target
    #                 edit_label = {'attribute': idx_to_attr[i], 'target_score': None, 'target_score_change': None}
    #             # edited_latent_code = None
    #             round_idx = 0
    #
    #             editing_logger.info(f'\ntarget attribute: {idx_to_attr[i]} ({i})')
    #             editing_logger.info(f'current cls: {attribute_dict_temp[idx_to_attr[i]]}')
    #             editing_logger.info(f'target cls: {target_scores[i]}')
    #
    #             attribute_dict_temp, pred_score_temp, exception_mode, latent_code, edited_latent_code, saved_image = edit_target_attribute(
    #                 opt, attribute_dict_temp, pred_score_temp, edit_label, round_idx, latent_code,
    #                 edited_latent_code, field_model, editing_logger,
    #                 print_intermediate_result)
    #
    #             if exception_mode != 'normal':
    #                 if exception_mode == 'already_at_target_class':
    #                     editing_logger.info(f"already_at_target_class. <Skip sample #{code_idx}>")
    #                     break  ############################
    #                 elif exception_mode == 'max_edit_num_reached':
    #                     editing_logger.info(f"max_edit_num_reached. <Skip sample #{code_idx}>")
    #                     break
    #                 elif exception_mode == 'current_class_not_clear':
    #                     editing_logger.info(f"current_class_not_clear. <Skip sample #{code_idx}>")
    #                     break
    #                 elif exception_mode == 'confidence_low':
    #                     editing_logger.info(f"confidence_low about edited result. <Skip sample #{code_idx}>")
    #                     break
    #
    #             # str_edit_types = '-'.join([idx_to_attr[k] for k in edit_idx_seq[:j+1]])
    #             str_edit_types = []
    #             for k in edit_idx_seq[:j+1]:
    #                 if idx_to_attr[k] == 'No_Beard':
    #                     str_edit_types.append('Beard')
    #                 else:
    #                     str_edit_types.append(idx_to_attr[k])
    #             str_edit_types = '-'.join(str_edit_types)
    #             save_image_path = os.path.join(opt["path"]["results_root"], f'{str_edit_types}')
    #             os.makedirs(save_image_path, exist_ok=True)
    #             if saved_image is not None:
    #                 save_image(saved_image, f'{save_image_path}/{code_idx:06d}.png')
    #
    #     if exception_mode == 'normal':
    #         good_list.append(code_idx)
    #
    # editing_logger.info('\n\n=============================')
    # editing_logger.info('finial good_list:')
    # editing_logger.info(good_list)

    
if __name__ == '__main__':
    main()

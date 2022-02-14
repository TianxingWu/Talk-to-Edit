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

    save_image_path = f'{opt["path"]["visualization"]}'
    os.makedirs(save_image_path, exist_ok=True)

    # ---------- create model ----------
    field_model = create_model(opt)

    # ---------- load latent code ----------
    random_codes = np.load(opt['latent_code_path'])
    os.makedirs(os.path.join(opt["path"]["results_root"], 'orig'), exist_ok=True)
    good_list = []

    target_scores = None
    target_score_changes = None

    if args.target_vals and not args.target_val_changes:
        target_scores = args.target_vals
    elif args.target_val_changes and not args.target_vals:
        target_score_changes = args.target_val_changes
    else:
        editing_logger.info("No input target_vals or target_val_changes, start binary editing.")

    # editing order
    edit_idx_seqs = np.array(opt['edit_idx_seqs'])

    # attributes = args.attrs
    attributes = ['Bangs', 'Eyeglasses', 'No_Beard', 'Smiling', 'Young']

    for code_idx in tqdm(range(len(random_codes))):
        editing_logger.info(f"================================================================== #{code_idx}")
        random_code = random_codes[code_idx, :]
        random_code = torch.from_numpy(random_code).to(torch.device('cuda'))
        with torch.no_grad():
            latent_code_orig = field_model.stylegan_gen.get_latent(random_code)
        latent_code_orig = latent_code_orig.cpu().numpy()

        # ---------- synthesize images ----------
        with torch.no_grad():
            start_image, start_label, start_score = \
                field_model.synthesize_and_predict(torch.from_numpy(latent_code_orig).to(torch.device('cuda')))  # noqa

        # select confident images
        # if np.any(np.array(start_score) < opt['confidence_thresh']):
        if np.any(np.array(start_score) < opt['confidence_thresh_input']):
            editing_logger.info(f"Not confident about original attribute class. <SKIP #{code_idx}>")
            continue

        # initialize attribute_dict
        attribute_dict = {
            "Bangs": start_label[0],
            "Eyeglasses": start_label[1],
            "No_Beard": start_label[2],
            "Smiling": start_label[3],
            "Young": start_label[4],
        }

        # restrictions ????????????????? or in base_model.py ?
        if attribute_dict['Bangs'] > 5 or \
            attribute_dict['Eyeglasses'] > 3 or \
            attribute_dict['No_Beard'] > 3 or \
            attribute_dict['Smiling'] > 4 or \
            attribute_dict['Young'] > 5:
            editing_logger.info(f"Restricted original attribute class. <Skip sample #{code_idx}>")
            continue

        # save_image(start_image, f'{opt["path"]["visualization"]}/start_image.png')
        save_image(start_image, os.path.join(opt["path"]["results_root"], 'orig', f'{code_idx:06d}.png'))

        target_scores = [None, None, None, None, None]
        if attribute_dict['Bangs'] == 0:
            # target_scores[0] = np.random.choice([1, 2, 3, 4, 5])
            target_scores[0] = np.random.choice([1, 2, 3])
        else:
            target_scores[0] = 0
        if attribute_dict['Eyeglasses'] == 0:
            target_scores[1] = np.random.choice([1, 2, 3])
            # target_scores[1] = np.random.choice([1, 2])
        else:
            target_scores[1] = 0
        if attribute_dict['No_Beard'] == 0:
            target_scores[2] = np.random.choice([1, 2, 3])
            # target_scores[2] = np.random.choice([1, 2])
        else:
            target_scores[2] = 0
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
        if attribute_dict['Young'] <= 2:
            target_scores[4] = np.random.choice([3, 4, 5])
        else:
            target_scores[4] = np.random.choice([0, 1, 2])

        print_intermediate_result = False
        attr_to_idx = opt['attr_to_idx']  # for print info

        for edit_idx_seq in edit_idx_seqs:
            latent_code = latent_code_orig.copy()
            edited_latent_code = None
            # import pdb
            # pdb.set_trace()
            attribute_dict_temp = attribute_dict.copy()  # attribute class changing
            editing_logger.info(f'\nstart_label: {start_label}, start_score: {start_score}')
            for j in range(len(edit_idx_seq)):
                i = edit_idx_seq[j]
                if target_scores:  # absolute target
                    edit_label = {'attribute': attributes[i], 'target_score': target_scores[i], 'target_score_change': None}
                elif target_score_changes:  # relative target
                    edit_label = {'attribute': attributes[i], 'target_score': None, 'target_score_change': target_score_changes[i]}
                else:  # binary target
                    edit_label = {'attribute': attributes[i], 'target_score': None, 'target_score_change': None}
                # edited_latent_code = None
                round_idx = 0

                editing_logger.info(f'\ntarget attribute: {attributes[i]} ({attr_to_idx[attributes[i]]})')
                editing_logger.info(f'current cls: {attribute_dict_temp[attributes[i]]}')
                editing_logger.info(f'target cls: {target_scores[i]}')

                attribute_dict_temp, exception_mode, latent_code, edited_latent_code, saved_image = edit_target_attribute(
                    opt, attribute_dict_temp, edit_label, round_idx, latent_code,
                    edited_latent_code, field_model, editing_logger,
                    print_intermediate_result)

                if exception_mode != 'normal':
                    if exception_mode == 'already_at_target_class':
                        editing_logger.info(f"already_at_target_class. <Skip sample #{code_idx}>")
                        break  ############################
                    elif exception_mode == 'max_edit_num_reached':
                        editing_logger.info(f"max_edit_num_reached. <Skip sample #{code_idx}>")
                        break
                    elif exception_mode == 'current_class_not_clear':
                        editing_logger.info(f"current_class_not_clear. <Skip sample #{code_idx}>")
                        break
                    elif exception_mode == 'confidence_low':
                        editing_logger.info(f"confidence_low about edited result. <Skip sample #{code_idx}>")
                        break

                # str_edit_types = '-'.join([attributes[k] for k in edit_idx_seq[:j+1]])
                str_edit_types = []
                for k in edit_idx_seq[:j+1]:
                    if attributes[k] == 'No_Beard':
                        str_edit_types.append('Beard')
                    else:
                        str_edit_types.append(attributes[k])
                str_edit_types = '-'.join(str_edit_types)
                save_image_path = os.path.join(opt["path"]["results_root"], f'{str_edit_types}')
                os.makedirs(save_image_path, exist_ok=True)
                if saved_image is not None:
                    save_image(saved_image, f'{save_image_path}/{code_idx:06d}.png')

        if exception_mode == 'normal':
            good_list.append(code_idx)

    editing_logger.info('\n\n=============================')
    editing_logger.info('finial good_list:')
    editing_logger.info(good_list)

    
if __name__ == '__main__':
    main()

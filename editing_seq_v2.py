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

    # import pdb
    # pdb.set_trace()
    # ---------- create model ----------
    field_model = create_model(opt)

    # ---------- load latent code ----------
    # if opt['inversion']['is_real_image']:
    #     latent_code = inversion(opt, field_model)
    # else:
    #
    #     if opt['latent_code_path'] is None:
    #         latent_code = torch.randn(1, 512, device=torch.device('cuda'))
    #         with torch.no_grad():
    #             latent_code = field_model.stylegan_gen.get_latent(latent_code)
    #         latent_code = latent_code.cpu().numpy()
    #         np.save(f'{opt["path"]["visualization"]}/latent_code.npz.npy',
    #                 latent_code)
    #     else:
    #         # ##############################
    #         latent_code = np.load(opt['latent_code_path'])
    #         # ##############################
    #         # i = opt['latent_code_index']
    #
    #         # latent_code = np.load(
    #         #     opt['latent_code_path'],
    #         #     allow_pickle=True).item()[f"{str(i).zfill(7)}.png"]
    #         # latent_code = torch.from_numpy(latent_code).to(
    #         #     torch.device('cuda'))
    #         # with torch.no_grad():
    #         #     latent_code = field_model.stylegan_gen.get_latent(latent_code)
    #         # latent_code = latent_code.cpu().numpy()
    random_codes = np.load(opt['latent_code_path'])

    os.makedirs('./results/orig', exist_ok=True)
    good_list = []

    target_scores = None
    target_score_changes = None
    if args.target_vals and not args.target_val_changes:
        target_scores = args.target_vals
    elif args.target_val_changes and not args.target_vals:
        target_score_changes = args.target_val_changes
    else:
        editing_logger.info("No input target_vals or target_val_changes, start binary editing.")

    for code_idx in tqdm(range(len(random_codes))):
        random_code = random_codes[code_idx, :]
        random_code = torch.from_numpy(random_code).to(torch.device('cuda'))
        with torch.no_grad():
            latent_code = field_model.stylegan_gen.get_latent(random_code)
        latent_code = latent_code.cpu().numpy()

        # ---------- synthesize images ----------
        with torch.no_grad():
            start_image, start_label, start_score = \
                field_model.synthesize_and_predict(torch.from_numpy(latent_code).to(torch.device('cuda')))  # noqa
                # field_model.synthesize_and_predict(latent_code) # noqa

        # save_image(start_image, f'{opt["path"]["visualization"]}/start_image.png')
        save_image(start_image, f'./results/orig/{code_idx}.png')


        editing_logger.info(f'\nstart_label: {start_label}, start_score: {start_score}')

        # initialize attribtue_dict
        attribute_dict = {
            "Bangs": start_label[0],
            "Eyeglasses": start_label[1],
            "No_Beard": start_label[2],
            "Smiling": start_label[3],
            "Young": start_label[4],
        }

        # edit_label = {'attribute': args.attr, 'target_score': args.target_val}
        attributes = args.attrs
        num_edit_types = len(attributes)

        # target_scores = None
        # target_score_changes = None
        # if args.target_vals and not args.target_val_changes:
        #     target_scores = args.target_vals
        # elif args.target_val_changes and not args.target_vals:
        #     target_score_changes = args.target_val_changes
        # else:
        #     editing_logger.info("No input target_vals or target_val_changes, start binary editing.")

        print_intermediate_result = False
        edited_latent_code = None
        attr_to_idx = opt['attr_to_idx']  # for print info
        for i in range(num_edit_types):
            if target_scores:  # absolute target
                edit_label = {'attribute': attributes[i], 'target_score': target_scores[i], 'target_score_change': None}
            elif target_score_changes:  # relative target
                edit_label = {'attribute': attributes[i], 'target_score': None, 'target_score_change': target_score_changes[i]}
            else:  # binary target
                edit_label = {'attribute': attributes[i], 'target_score': None, 'target_score_change': None}
            # edited_latent_code = None
            round_idx = 0

            editing_logger.info(f'\ntarget attribute: {attributes[i]} ({attr_to_idx[attributes[i]]})')
            editing_logger.info(f'current cls: {attribute_dict[attributes[i]]}')
            # import pdb
            # pdb.set_trace()
            attribute_dict, exception_mode, latent_code, edited_latent_code, saved_image = edit_target_attribute(
                opt, attribute_dict, edit_label, round_idx, latent_code,
                edited_latent_code, field_model, editing_logger,
                print_intermediate_result)

            if exception_mode != 'normal':
                if exception_mode == 'already_at_target_class':
                    editing_logger.info("This attribute is already at the degree that you want. Let's try a different attribute degree or another attribute.")
                    break  ############################
                elif exception_mode == 'max_edit_num_reached':
                    editing_logger.info(f"Sorry, we are unable to edit this attribute. Perhaps we can try something else. [Skip code #{code_idx}]")
                    break
                elif exception_mode == 'current_class_not_clear':
                    editing_logger.info(f"Current class not clear. [Skip code #{code_idx}]")
                    break

            str_edit_types = '-'.join(attributes[:i+1])
            save_image_path = f'./results/{str_edit_types}'
            os.makedirs(save_image_path, exist_ok=True)
            if saved_image is not None:
                save_image(saved_image, f'{save_image_path}/{code_idx}.png')

        if exception_mode == 'normal':
            good_list.append(code_idx)

    editing_logger.info('\n\n=============================')
    editing_logger.info('finial good_list:')
    editing_logger.info(good_list)

    
if __name__ == '__main__':
    main()

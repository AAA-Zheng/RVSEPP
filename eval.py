import os
import argparse
import logging
import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', help='coco or f30k')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--cxc_annot_base', default='')
    parser.add_argument('--save_results', default=True)
    opt = parser.parse_args()

    weights_bases = ['']

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
        else:
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=True)
            # Evaluate COCO 5K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    main()

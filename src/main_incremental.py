import logging, os, time, torch, argparse
import numpy as np
from rich.logging import RichHandler
from networks.extractor_ensemble import ExtractorEnsemble
import utils
from functools import reduce
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from torchvision import models
from approach.seed import SeedAppr
from approach.joint import JointAppr

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)

log = logging.getLogger("rich")


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='DivideAndNotForget - Berga - Righetti')

    # miscellaneous args
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=['cifar100', 'food101'],
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=['resnet18', 'resnet50', 'resnet101'],
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=1, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--approach', default='seed', type=str, choices=['seed', 'joint'],
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    
    # To understand
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, fix_bn=args.fix_bn)
    
    SEED = 99
    GPU = 0
    NC_FIRST_TASK=10
    utils.seed_everything(seed=SEED)
    
    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU)
        device = 'cuda'
        log.info('[bold green blink]Cuda available, using GPU! [/]')
    else:
        log.error('[bold red blink]WARNING: [CUDA unavailable] Using CPU instead![/]')
        device = 'cpu'

    ###### NETWORK LOADING #######
    if args.network == 'resnet18':
        init_model = models.resnet18(weights='IMAGENET1K_V1')
    elif args.network == 'resnet50':
        init_model = models.resnet50(weights='IMAGENET1K_V1')
    elif args.network == 'resnet101':
        init_model = models.resnet101(weights='IMAGENET1K_V1')
    # Set explicitly classifier variable in Resnet models
    init_model.head_var = 'fc'

    # ###### CONTINUAL LEARNING APPROACH (SEED) #######
    if args.approach == 'seed':
        approach = SeedAppr
    elif args.approach == 'joint':
        approach = JointAppr
    appr_args, extra_args = approach.extra_parser(extra_args)
    log.info("[blue]Using {app} approach[/]".format(app=args.approach))

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    # ###### Instantiate the multilogger #######
    logger = MultiLogger(args.results_path, full_exp_name, loggers=['disk'], save_models=args.save_models)

    # SEED everything to reprodicibility
    utils.seed_everything(seed=SEED)
    
    # ###### Generate the data loaders, one for each task #######
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, 
                                                              NC_FIRST_TASK, args.batch_size)
    max_task = len(taskcla)

    # Network and Approach instances
    utils.seed_everything(seed=SEED)

    # ###### Instantiate the complete model for training #######
    # intit_model is used as backbone
    net = ExtractorEnsemble(init_model, taskcla, args.network, device)
   
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=log, **appr_args.__dict__)}

    utils.seed_everything(seed=SEED)
    appr = approach(net, device, **appr_kwargs)


    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        log.info('*' * 108)
        log.info('Task {:2d}'.format(t))
        log.info('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # Train
        appr.train(t, trn_loader[t], val_loader[t])
        log.info('-' * 108)

        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])

            # Save
            print('Save at ' + os.path.join(args.results_path, full_exp_name))
            logger.log_result(acc_taw, name="acc_taw", step=t)
            logger.log_result(acc_tag, name="acc_tag", step=t)
            # logger.log_result(forg_taw, name="forg_taw", step=t)
            # logger.log_result(forg_tag, name="forg_tag", step=t)
            logger.save_model(net.state_dict(), task=t)
            logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
            logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
            aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
            logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
            logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()

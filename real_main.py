import argparse
import yaml
import os
from datetime import datetime
import logging
import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import wandb

from src.utils import set_seed, freeze_model, melt_model
from src.data import (get_retain_forget_datasets, get_dataloaders, get_train_test_datasets,
                      get_transforms, get_exact_surr_datasets)
from src.loss import L2RegularizedCrossEntropyLoss
from src.train import train
from src.eval import evaluate
from src.forget import forget, sample_from_exact_marginal, estimate_marginal_kl_distance
from src.metrics import membership_inference_attack, relearn_time

from torch.utils.data import Subset

def log_eval(model, train_loader, val_loader, retain_loader, forget_loader, criterion, device, dss_loader=None):
    train_acc = evaluate(train_loader, model, criterion, device=device, log=True)
    test_acc = evaluate(val_loader, model, criterion, device=device, log=True)
    retain_acc = evaluate(retain_loader, model, criterion, device=device, log=True)
    forget_acc = evaluate(forget_loader, model, criterion, device=device, log=True)
    
    metrics = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'retain_acc': retain_acc,
        'forget_acc': forget_acc
    }
    
    # evaluate on D_ss subset
    if dss_loader is not None:
        dss_acc = evaluate(dss_loader, model, criterion, device=device, log=True)
        metrics['dss_acc'] = dss_acc
        logging.info(
            'train: {}, test: {}, retain: {}, forget: {}, D_ss: {}'.format(train_acc, test_acc, retain_acc, forget_acc, dss_acc))
    else:
        logging.info(
            'train: {}, test: {}, retain: {}, forget: {}'.format(train_acc, test_acc, retain_acc, forget_acc))
    
    # TODO: mais Old surrogate evaluation
    # surr_acc = evaluate(surr_loader, model, criterion, device=device, log=True)
    # logging.info(
    #     'train: {}, test: {}, retain: {}, forget: {}, surrogate:{}'.format(train_acc, test_acc, retain_acc, forget_acc,
    #                                                                        surr_acc))
    return metrics


def return_model(model_config, dim, num_class):
    if model_config['type'] == 'mlp':
        bias = model_config['bias']
        if model_config['hidden_sizes'] is not None:
            model_arr = [nn.Flatten()]  # Flatten input images first # it was []
            curr_in = dim
            for size in model_config['hidden_sizes']:
                model_arr.append(nn.Linear(curr_in, size, bias=bias))
                if model_config['activation'] == 'relu':
                    model_arr.append(nn.ReLU())
                curr_in = size
            model_arr.append(nn.Linear(curr_in, num_class, bias=bias))
            model = nn.Sequential(*model_arr)
        else:
            model = nn.Sequential(nn.Flatten(), nn.Linear(dim, num_class, bias=bias)) # 
        return model
    elif model_config['type'] == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if model_config['mode'] == 'linear':
            model = nn.Sequential(nn.Flatten(),
                                  nn.Linear(model.fc.in_features, num_class))
        elif model_config['mode'] == 'conv1':
            model = nn.Sequential(
                model.layer4[1],  # Fourth residual block
                model.avgpool,  # Global average pooling
                nn.Flatten(),  # Flatten the tensor
                nn.Linear(model.fc.in_features, num_class)  # Fully connected layer
            )

            for idx, param in enumerate(model.parameters()):
                param.requires_grad = False
                if idx == 2:
                    break
        elif model_config['mode'] == 'conv2':
            model = nn.Sequential(
                model.layer4[1],  # Fourth residual block
                model.avgpool,  # Global average pooling
                nn.Flatten(),  # Flatten the tensor
                nn.Linear(model.fc.in_features, num_class)  # Fully connected layer
            )
        return model


def replace_none_with_none(d):
    for k, v in d.items():
        if isinstance(v, dict):
            replace_none_with_none(v)
        elif v == 'none':
            d[k] = None


def main():
    parser = argparse.ArgumentParser(description='wrapper of all real dataset experiments')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    # read config set experiment
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    replace_none_with_none(config)
    base_save_dir = config['setup']['base_save_dir']
    about = config['setup']['about']
    curr_dict = config
    for key in about.split('-'):
        curr_dict = curr_dict[key]
    about_value = curr_dict
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    now = datetime.now()
    experiment_dir = os.path.join(base_save_dir, '{}-{}-{}'.format(about,
                                                                   str(about_value),
                                                                   now.strftime('%Y-%m-%d-%H-%M-%S')))
    os.makedirs(experiment_dir)
    logging.basicConfig(filename=os.path.join(experiment_dir, 'experiment.log'), level=logging.INFO)
    logging.info('experiment started at %s', now.strftime('%Y-%m-%d %H:%M:%S'))

    # Initialize wandb
    wandb_config = config['setup'].get('wandb', {})
    if wandb_config.get('enabled', False):
        wandb.init(
            project=wandb_config.get('project', 'certified-unlearning'),
            entity=wandb_config.get('entity', None),
            config=config,
            name='{}-{}'.format(about, str(about_value)),
            dir=experiment_dir,
            tags=['subsampled-hessian']
        )
        logging.info('wandb initialized: project={}, entity={}'.format(
            wandb_config.get('project'), wandb_config.get('entity', 'default')))
    else:
        wandb.init(mode='disabled')
        logging.info('wandb disabled')

    # copy config
    config_copy_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_copy_path, 'w') as file:
        yaml.safe_dump(config, file)
    logging.info('configuration file copied to %s', config_copy_path)

    # set seed
    seed = config['setup']['seed']
    set_seed(seed)
    logging.info('seed: %s', seed)

    # set device
    device_id = config['setup']['device']
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    logging.info('device: %s', device)

    # set data
    logging.info('#####################')
    logging.info('setting data')
    data_config = config['data']
    exact_dataset_type = data_config['exact_dataset']
    # TODO: mais - Old surrogate dataset approach
    # surrogate_dataset_type = data_config['surrogate_dataset']
    dim = data_config['dim']
    num_class = data_config['num_class']
    model_type = config['train']['model']['type']
    transforms = get_transforms(exact_dataset_type, model_type=model_type)
    train_dataset, test_dataset = get_train_test_datasets(exact_dataset_type, transform=transforms,
                                                         train_path=data_config['train_path'],
                                                         test_path=data_config['test_path'],
                                                         save_path=data_config['save_path'],
                                                         device=device)
    
    # TODO: mais COMMENTED Old surrogate dataset creation - not needed for subsampled-Hessian method
    # if surrogate_dataset_type is not None:
    #     stransforms = get_transforms(surrogate_dataset_type)
    #     surr_dataset, _ = get_train_test_datasets(surrogate_dataset_type, transform=transforms,
    #                                               train_path=data_config['strain_path'],
    #                                               test_path=data_config['stest_path'],
    #                                               save_path=data_config['ssave_path'],
    #                                               device=device)
    # else:
    #     exact_size = int(len(train_dataset) / 2)
    #     surr_size = len(train_dataset) - exact_size
    #     dirichlet = data_config['dirichlet']
    #     train_dataset, surr_dataset = get_exact_surr_datasets(train_dataset,
    #                                                           target_size=exact_size,
    #                                                           starget_size=surr_size,
    #                                                           dirichlet=dirichlet, num_class=num_class)

    # NEW: Create subset D_ss for subsampled-Hessian computation
    # D_ss is a subset of the training data used for Hessian approximation
    subsample_ratio = config['unlearn'].get('subsample_ratio', 0.5)
    dss_size = int(len(train_dataset) * subsample_ratio)
    dss_indices = torch.randperm(len(train_dataset))[:dss_size].tolist()
    
    dss_dataset = Subset(train_dataset, dss_indices)
    
    logging.info('training dataset and D_ss subset created')
    logging.info('full training dataset size: {}, dim: {}'.format(len(train_dataset), dim))
    logging.info('D_ss subset size: {} ({:.1%} of training data)'.format(len(dss_dataset), subsample_ratio))
    # mais COMMENTED surrogate dataset logging
    # logging.info('surrogate dataset size: {}, dim: {}'.format(len(surr_dataset), dim))
    logging.info('#####################')

    logging.info('#####################')
    logging.info('training setup')
    # train
    train_config = config['train']

    # set train test data
    forget_ratio = config['unlearn']['forget_ratio']
    retain_dataset, forget_dataset = get_retain_forget_datasets(train_dataset, forget_ratio)
    train_loader, test_loader = get_dataloaders([train_dataset, test_dataset], train_config['batch_size'])
    retain_loader, forget_loader = get_dataloaders([retain_dataset, forget_dataset], train_config['batch_size'])
    # NEW: Create loader for D_ss subset (used for Hessian computation)
    dss_loader = get_dataloaders(dss_dataset, train_config['batch_size'])
    # COMMENTED: Old surrogate loader - not needed for subsampled-Hessian method
    # surr_loader = get_dataloaders(surr_dataset, train_config['batch_size'])

    logging.info('all dataloaders created')
    logging.info(
        'forget ratio: {}, retain dataset size: {}, forget dataset size: {}'.format(forget_ratio, len(retain_dataset),
                                                                                    len(forget_dataset)))
    logging.info('D_ss subset loader created with {} samples'.format(len(dss_dataset)))

    lambda_param = train_config['lambda']
    criterion = L2RegularizedCrossEntropyLoss(lambda_param)
    logging.info('criterion: {}, lambda: {}'.format(criterion, lambda_param))

    # set model
    model_config = train_config['model']
    model = return_model(model_config, dim, num_class)
    model = model.to(device)
    logging.info('model: {}'.format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    logging.info('lr: {}'.format(train_config['lr']))
    logging.info('#####################')

    langevin_config = config['langevin']

    # train
    num_epochs = train_config['num_epochs']
    logging.info('#####################')
    logging.info('INITIAL TRAINING')
    train(train_loader, test_loader, model, criterion, optimizer, num_epoch=num_epochs, device=device, log_prefix='original/')
    metrics = log_eval(model, train_loader, test_loader, retain_loader, forget_loader, criterion, device, dss_loader=dss_loader)
    target_acc = metrics['forget_acc']
    
    # Log to wandb
    wandb.log({
        'original/train_acc': metrics['train_acc'],
        'original/test_acc': metrics['test_acc'],
        'original/retain_acc': metrics['retain_acc'],
        'original/forget_acc': metrics['forget_acc'],
    })
    if 'dss_acc' in metrics:
        wandb.log({'original/dss_acc': metrics['dss_acc']})
    
    # COMMENTED: Langevin sampling - not needed for subsampled-Hessian method
    # egensample_loader = sample_from_exact_marginal(model, langevin_config['num_samples'],
    #                                                langevin_config['input_size'],
    #                                                train_config['batch_size'],
    #                                                input_range=langevin_config['input_range'],
    #                                                max_iter=langevin_config['max_iter'],
    #                                                step_size=langevin_config['step_size'])
    
    model = model.to('cpu')
    # save model state dict
    model_save_path = os.path.join(experiment_dir, 'initial_model.pth')
    torch.save(model.state_dict(), model_save_path)
    logging.info('initial model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # COMMENTED: Old surrogate model training - not needed for subsampled-Hessian method
    # # surrogate training
    # logging.info('#####################')
    # logging.info('SURROGATE MODEL TRAINING')
    # smodel = return_model(model_config, dim, num_class)
    # smodel = smodel.to(device)
    # optimizer = torch.optim.Adam(smodel.parameters(), lr=train_config['lr'])
    # train(surr_loader, test_loader, smodel, criterion, optimizer, num_epoch=num_epochs, device=device)
    # log_eval(smodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    # sgensample_loader = sample_from_exact_marginal(smodel, langevin_config['num_samples'],
    #                                                langevin_config['input_size'],
    #                                                train_config['batch_size'],
    #                                                input_range=langevin_config['input_range'],
    #                                                max_iter=langevin_config['max_iter'],
    #                                                step_size=langevin_config['step_size'])
    # smodel = smodel.to('cpu')
    # model_save_path = os.path.join(experiment_dir, 'surrogate_model.pth')
    # torch.save(smodel.state_dict(), model_save_path)
    # logging.info('surrogate model state dict saved to %s', model_save_path)
    # logging.info('#####################')

    # COMMENTED: KL distance estimation - not needed for subsampled-Hessian method
    # # kl distance estimation
    # _, kl_distance = estimate_marginal_kl_distance(sgensample_loader, egensample_loader, device)
    # _.to('cpu')
    # del _
    # logging.info('kl distance estimated using generated samples is {}'.format(kl_distance))

    # retrain
    logging.info('#####################')
    logging.info('RETRAIN FROM SCRATCH')
    rmodel = return_model(model_config, dim, num_class)
    rmodel = rmodel.to(device)
    optimizer = torch.optim.Adam(rmodel.parameters(), lr=train_config['lr'])
    train(retain_loader, test_loader, rmodel, criterion, optimizer, num_epoch=num_epochs, device=device, log_prefix='retrained/')
    metrics = log_eval(rmodel, train_loader, test_loader, retain_loader, forget_loader, criterion, device, dss_loader=dss_loader)
    mia_score = membership_inference_attack(rmodel, test_loader, forget_loader)
    logging.info('MIA {}'.format(mia_score))
    required_iters = relearn_time(rmodel, criterion, forget_loader, lr=train_config['lr'],
                                  target_acc=target_acc)
    logging.info('relearn time T {}'.format(required_iters))
    
    # Log to wandb
    wandb.log({
        'retrained/train_acc': metrics['train_acc'],
        'retrained/test_acc': metrics['test_acc'],
        'retrained/retain_acc': metrics['retain_acc'],
        'retrained/forget_acc': metrics['forget_acc'],
        'retrained/mia_score': mia_score,
        'retrained/relearn_iters': required_iters,
    })
    if 'dss_acc' in metrics:
        wandb.log({'retrained/dss_acc': metrics['dss_acc']})
    
    rmodel = rmodel.to('cpu')
    model_save_path = os.path.join(experiment_dir, 'retrained_model.pth')
    torch.save(rmodel.state_dict(), model_save_path)
    logging.info('retrained model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # unlearn config
    unlearn_config = config['unlearn']
    forget_ratio = unlearn_config['forget_ratio']
    eps_multiplier = unlearn_config['eps_multiplier']
    eps_power = unlearn_config['eps_power']
    delta = unlearn_config['delta']
    
    # TODO: mais subsampled-Hessian parameters from config
    eta = unlearn_config.get('eta', 0.01)
    alpha = unlearn_config.get('alpha', 1.0)
    beta = unlearn_config.get('beta', 1.0)
    gamma = unlearn_config.get('gamma', 1.0)
    L = unlearn_config.get('L', 1.0)
    C_constant = unlearn_config.get('C_constant', 2.0)
    
    # Old parameters (may not be used in new algorithm)
    smooth = unlearn_config.get('smooth', 1)
    sc = unlearn_config.get('sc', 1)
    lip = unlearn_config.get('lip', 1)
    hlip = unlearn_config.get('hlip', 1)
    
    # Computation options
    linear = unlearn_config.get('linear', False)
    parallel = unlearn_config.get('parallel', False)
    cov = unlearn_config.get('cov', False)
    conjugate = unlearn_config.get('conjugate', False)
    
    # Method selection
    use_subsampled = unlearn_config.get('subsampled', False)

    # COMMENTED: Old unlearn with exact approach
    # # unlearn with exact
    # logging.info('#####################')
    # logging.info('UNLEARN WITH EXACT')
    # logging.info('noise --> eps_multiplier: {}, eps_power: {}, delta: {}, smooth: {}, sc: {}, lip: {}, hlip: {}'.format(
    #     eps_multiplier, eps_power, delta, smooth, sc, lip, hlip))
    # eps = eps_multiplier * (math.e ** eps_power)
    # umodel = forget(model, train_loader, forget_loader, forget_loader, criterion, device, save_path=experiment_dir,
    #                 eps=eps, delta=delta, smooth=smooth, sc=sc, lip=lip, hlip=hlip, linear=linear,
    #                 parallel=parallel, cov=cov, alpha=alpha, conjugate=conjugate)
    # log_eval(umodel, train_loader, test_loader, retain_loader, forget_loader, criterion, device, dss_loader=dss_loader)
    # mia_score = membership_inference_attack(umodel, test_loader, forget_loader)
    # logging.info('MIA {}'.format(mia_score))
    # required_iters = relearn_time(umodel, criterion, forget_loader, lr=train_config['lr'],
    #                               target_acc=target_acc)
    # logging.info('relearn time T {}'.format(required_iters))
    # umodel = umodel.to('cpu')
    # model_save_path = os.path.join(experiment_dir, 'uexact_model.pth')
    # torch.save(umodel.state_dict(), model_save_path)
    # logging.info('unlearn with exact model state dict saved to %s', model_save_path)
    # logging.info('#####################')

    # Subsampled-Hessian Unlearning
    if use_subsampled:
        from src.forget import subsampled_hessian_unlearning
        
        logging.info('#####################')
        logging.info('SUBSAMPLED-HESSIAN UNLEARNING')
        
        # Calculate dataset sizes
        n1 = len(train_dataset)  # Total training size
        n2 = len(dss_dataset)    # D_ss subset size
        m = len(forget_dataset)  # Forget set size
        
        # Calculate number of model parameters (d)
        d = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info('n1={}, n2={}, m={}, d={}'.format(n1, n2, m, d))
        logging.info('Parameters: eps={}, delta={}, eta={}'.format(eps_multiplier * (math.e ** eps_power), delta, eta))
        logging.info('Regularity: alpha={}, beta={}, gamma={}, L={}'.format(alpha, beta, gamma, L))
        logging.info('Hessian method: {}'.format('Analytical (linearized)' if linear else 'Conjugate Gradient (implicit)'))
        
        eps = eps_multiplier * (math.e ** eps_power)
        
        umodel = subsampled_hessian_unlearning(
            model=model,
            dss_loader=dss_loader,  # D_ss for Hessian computation
            forget_loader=forget_loader,  # D_u for unlearning
            criterion=criterion,
            device=device,
            n1=n1,          
            n2=n2,         
            m=m,            
            eps=eps,        
            delta=delta,    
            eta=eta,        
            alpha=alpha,    
            beta=beta,      
            gamma=gamma,    
            L=L,            
            d=d,           
            C_constant=C_constant,  
            linear=linear   # Use analytical Hessian (True) or conjugate gradient (False)
        )
        
        metrics = log_eval(umodel, train_loader, test_loader, retain_loader, forget_loader, criterion, device, dss_loader=dss_loader)
        mia_score = membership_inference_attack(umodel, test_loader, forget_loader)
        logging.info('MIA {}'.format(mia_score))
        required_iters = relearn_time(umodel, criterion, forget_loader, lr=train_config['lr'],
                                      target_acc=target_acc)
        logging.info('relearn time T {}'.format(required_iters))
        
        # Log to wandb
        wandb.log({
            'unlearned/train_acc': metrics['train_acc'],
            'unlearned/test_acc': metrics['test_acc'],
            'unlearned/retain_acc': metrics['retain_acc'],
            'unlearned/forget_acc': metrics['forget_acc'],
            'unlearned/mia_score': mia_score,
            'unlearned/relearn_iters': required_iters,
            'hyperparams/eps': eps,
            'hyperparams/delta': delta,
            'hyperparams/eta': eta,
            'hyperparams/alpha': alpha,
            'hyperparams/beta': beta,
            'hyperparams/gamma': gamma,
            'hyperparams/subsample_ratio': subsample_ratio,
        })
        if 'dss_acc' in metrics:
            wandb.log({'unlearned/dss_acc': metrics['dss_acc']})
        
        umodel = umodel.to('cpu')
        model_save_path = os.path.join(experiment_dir, 'subsampled_hessian_model.pth')
        torch.save(umodel.state_dict(), model_save_path)
        logging.info('subsampled-Hessian unlearned model saved to %s', model_save_path)
        logging.info('#####################')

    # COMMENTED: Old surrogate unlearning - not needed for subsampled-Hessian method
    # if surr:
    #     # unlearn with surrogate
    #     logging.info('#####################')
    #     logging.info('UNLEARN WITH SURROGATE')
    #     logging.info(
    #         'noise --> eps_multiplier: {}, eps_power: {}, delta: {}, smooth: {}, sc: {}, lip: {}, hlip: {}, kl_distance: {}'.format(
    #             eps_multiplier, eps_power, delta, smooth, sc, lip, hlip, kl_distance))
    #     smodel = smodel.to(device)
    #     usmodel = forget(model, surr_loader, forget_loader, forget_loader, criterion, device, save_path=experiment_dir,
    #                      eps=eps, delta=delta, smooth=smooth, sc=sc, lip=lip, hlip=hlip, surr=surr,
    #                      known=known, surr_loader=surr_loader, surr_model=smodel, kl_distance=kl_distance,
    #                      linear=linear, parallel=parallel, cov=cov, alpha=alpha, conjugate=conjugate, prev_size=len(train_dataset))
    #     log_eval(usmodel, train_loader, test_loader, retain_loader, forget_loader, criterion, device, dss_loader=dss_loader)
    #     smodel = smodel.to('cpu')
    #     mia_score = membership_inference_attack(usmodel, test_loader, forget_loader)
    #     logging.info('MIA {}'.format(mia_score))
    #     required_iters = relearn_time(usmodel, criterion, forget_loader, lr=train_config['lr'],
    #                                   target_acc=target_acc)
    #     logging.info('relearn time T {}'.format(required_iters))
    #     usmodel = usmodel.to('cpu')
    #     model_save_path = os.path.join(experiment_dir, 'usurr_model.pth')
    #     torch.save(usmodel.state_dict(), model_save_path)
    #     logging.info('unlearn with surrogate model state dict saved to %s', model_save_path)
    #     logging.info('#####################')
    
    # Finish wandb run
    wandb.finish()
    logging.info('experiment completed')


if __name__ == '__main__':
    main()

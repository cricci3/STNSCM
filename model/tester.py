import os
import torch
import numpy as np
from tqdm import tqdm
from tools.metrics import metric
from model.helper import SimpleTrainer, Trainer


def model_val(runid, engine, dataloader, device, logger, epoch):
    logger.info('Start validation phase.....')
    val_loader = dataloader['val']

    loss_list, mae_list, mape_list, rmse_list, preds = [], [], [], [], []

    for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        if len(batch) >= 6:
            x, x_time, target, target_time, pos, target_cl = batch
        else:
            x, target = batch
            x_time = target_time = pos = target_cl = None

        x = x.to(device)
        target = target.to(device)

        loss, mae, mape, rmse, pred = engine.eval(input=x, target=target)

        loss_list.append(loss)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        preds.append(pred)

    logger.info(f"[Val] Loss: {np.mean(loss_list):.4f}, MAE: {np.mean(mae_list):.4f}, MAPE: {np.mean(mape_list):.4f}, RMSE: {np.mean(rmse_list):.4f}")
    return np.mean(loss_list), np.mean(mae_list), np.mean(mape_list), np.mean(rmse_list), torch.cat(preds)

def model_test(runid, engine, dataloader, device, logger, cfg, mode='Test'):
    logger.info('Start testing phase.....')
    test_loader = dataloader['test']

    loss_list, mae_list, mape_list, rmse_list, preds, targets = [], [], [], [], [], []

    for _, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        if len(batch) >= 6:
            x, x_time, target, target_time, pos, target_cl = batch
        else:
            x, target = batch
            x_time = target_time = pos = target_cl = None

        x = x.to(device)
        target = target.to(device)

        loss, mae, mape, rmse, pred = engine.eval(input=x, target=target)

        loss_list.append(loss)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        preds.append(pred)
        targets.append(target)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    logger.info(f"[Test] Loss: {np.mean(loss_list):.4f}, MAE: {np.mean(mae_list):.4f}, MAPE: {np.mean(mape_list):.4f}, RMSE: {np.mean(rmse_list):.4f}")

    return np.mean(loss_list), np.mean(mae_list), np.mean(mape_list), np.mean(rmse_list), preds


def baseline_test(runid, model, dataloader, device, logger, cfg):
    scaler = dataloader['scaler']

    # Trainer adattivo
    if 'dummy' in cfg['model_name'].lower():
        engine = SimpleTrainer(
            model=model,
            lr=cfg['train']['base_lr'],
            weight_decay=cfg['train']['weight_decay'],
            loss_type=cfg['model']['loss_type'],
            scaler=scaler,
            device=device
        )
    else:
        engine = Trainer(
            model,
            base_lr=cfg['train']['base_lr'],
            weight_decay=cfg['train']['weight_decay'],
            milestones=cfg['train']['milestones'],
            lr_decay_ratio=cfg['train']['lr_decay_ratio'],
            min_learning_rate=cfg['train']['min_learning_rate'],
            max_grad_norm=cfg['train']['max_grad_norm'],
            cl_decay_steps=cfg['train']['cl_decay_steps'],
            num_for_target=cfg['data']['num_for_target'],
            num_for_predict=cfg['data']['num_for_predict'],
            loss_type=cfg['model']['loss_type'],
            scaler=scaler,
            device=device,
            curriculum_learning=cfg['train']['use_curriculum_learning'],
            new_training=cfg['train']['new_training'],
        )

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path, map_location=torch.device('mps'), weights_only=False)
    engine.model.load_state_dict(save_dict['model_state_dict'], strict=False)
    logger.info('model load success! {}'.format(best_mode_path))

    # 计算参数数量
    total_param = 0
    logger.info('Net\'s state_dict:')
    for param_tensor in engine.model.state_dict():
        logger.info(param_tensor + '\t' + str(engine.model.state_dict()[param_tensor].size()))
        total_param += np.prod(engine.model.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{:d}'.format(int(total_param)))

    logger.info('Optimizer\'s state_dict:')
    for var_name in engine.optimizer.state_dict():
        logger.info(var_name + '\t' + str(engine.optimizer.state_dict()[var_name]))

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('Number of model parameters is {:d}'.format(int(nParams)))

    mtest_loss, mtest_mae, mtest_mape, mtest_rmse, predicts = model_test(runid, engine, dataloader, device, logger,
                                                                         cfg, mode='Test')
    return mtest_mae, mtest_mape, mtest_rmse, mtest_mae, mtest_mape, mtest_rmse
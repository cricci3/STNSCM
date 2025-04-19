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

    # Load best model if specified
    if cfg['train'].get('best_mode', None):
        best_path = cfg['train']['best_mode']
        logger.info(f"Loading checkpoint from {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        engine.model.load_state_dict(checkpoint['model_state_dict'])

    return model_test(runid, engine, dataloader, device, logger, cfg)
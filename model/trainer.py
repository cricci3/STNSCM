import copy
import time
import torch
import numpy as np
from tqdm import tqdm
import os

from model.tester import model_val, model_test

def baseline_train(runid, model, model_name, dataloader, static_norm_adjs, device, logger, cfg):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    logger.info("Start training...")

    save_path = os.path.join(cfg['save'], cfg['model_name'], cfg['data']['freq'], 'ckpt')
    os.makedirs(save_path, exist_ok=True)
    scaler = dataloader['scaler']

    # Selezione dinamica del Trainer
    if 'dummy' in cfg['model_name'].lower():
        from model.helper import SimpleTrainer as Trainer


        engine = Trainer(
            model=model,
            lr=cfg['train']['base_lr'],
            weight_decay=cfg['train']['weight_decay'],
            loss_type=cfg['model']['loss_type'],
            scaler=scaler,
            device=device
        )
    else:
        from model.helper import Trainer
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

    # Setup training loop
    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']
    print_every = cfg['train']['print_every']

    best_val_loss = float('inf')
    stable_count = 0
    best_model = None
    global_step = 0
    his_loss = []
    train_time, val_time = [], []

    for epoch in range(begin_epoch, begin_epoch + epochs + 1):
        train_loss, train_mae, train_mape, train_rmse = [], [], [], []
        t1 = time.time()
        train_loader = dataloader['train']

        for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Adatta al formato dei modelli dummy
            if len(batch) >= 6:
                x, x_time, target, target_time, pos, target_cl = batch
            else:
                x, target = batch
                x_time = target_time = pos = target_cl = None

            x = x.to(device)
            target = target.to(device)

            metrics = engine.train(input=x, target=target)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            global_step += 1

        t2 = time.time()
        train_time.append(t2 - t1)

        # VALIDATION
        s1 = time.time()
        valid_loss, valid_mae, valid_mape, valid_rmse, _ = model_val(
            runid, engine, dataloader, device, logger, epoch
        )
        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(valid_loss)

        logger.info(f"Epoch {epoch:03d}, Train Loss: {mtrain_loss:.4f}, Valid Loss: {mvalid_loss:.4f}")

        his_loss.append(mvalid_loss)
        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            stable_count = 0
            best_model = copy.deepcopy(engine.model.state_dict())
            ckpt_name = f"exp{model_name}_epoch{epoch}_ValLoss:{mvalid_loss:.4f}.pth"
            best_path = os.path.join(save_path, ckpt_name)
            torch.save({'model_state_dict': best_model}, best_path)
            logger.info(f"Better model saved: {best_path}")
        else:
            stable_count += 1
            if stable_count > tolerance:
                logger.info("Early stopping triggered.")
                break

    # LOAD BEST MODEL
    engine.model.load_state_dict(best_model)
    logger.info("Training completed. Evaluating on test set...")

    valid_loss, valid_mae, valid_mape, valid_rmse, _ = model_val(
        runid, engine, dataloader, device, logger, epoch
    )
    test_loss, test_mae, test_mape, test_rmse, _ = model_test(
        runid, engine, dataloader, device, logger, cfg
    )

    return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse
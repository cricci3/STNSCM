#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
==================================================================
               **Life is short, You need Python!**
File         : main.py
Project      : AAAI2023-STNSCN
Created Date : 2021/11/14 22:17
Author       : Yu Zhao
Email        : yzhao@buaa.edu.cn
==================================================================
Descriptions :
==================================================================
TODO List:
   Date                    Comments                        Finish
---------   --------------------------------------------   -------
"""
from datetime import datetime
import torch
import numpy as np
import sys
import os
from model.STNSCN import STNSCN
from GRU import DummyGRU
from AGCRN_model.AGCRN import AGCRN
from model.tester import baseline_test
from model.trainer import baseline_train
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from config.config import get_logger
from model.datasets import load_dataset
from tools.utils import asym_adj
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == "__main__":
    data_name = "NYC"
    
    # Chiedi all'utente quale modello vuole utilizzare
    name_model = input("Select the model u want to use (S,G,A): ").upper()
    while name_model not in ["A", "G", "S"]:
        name_model = input("Errore: Select the model u want to use (S,G,A): ").upper()
    
    # Scegli il file di configurazione appropriato in base al modello
    if name_model == "S":
        print("STNSC selected")
        config_filename = "data/ckpt/" + data_name + "/configSTNSCN.yaml"
    elif name_model == "G":
        print("DummyGRU selected")
        config_filename = "data/ckpt/" + data_name + "/configDummy.yaml"
    elif name_model == "A":
        print("AGCRN selected")
        config_filename = "data/ckpt/" + data_name + "/configAGCRN.yaml"
    
    # Carica il file di configurazione
    with open(config_filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    
    # Impostazione dei percorsi e setup generale
    base_path = cfg["base_path"]
    dataset_name = cfg["dataset_name"]
    dataset_path = os.path.join(base_path, dataset_name)
    
    # Log Path
    log_path = os.path.join("data/ckpt/" + data_name, cfg["model_name"], cfg["data"]["freq"], "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    
    # Result Save Path
    save_path = os.path.join("data/ckpt/" + data_name, cfg["model_name"], cfg["data"]["freq"], "ckpt")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Log
    log_dir = log_path
    log_level = "INFO"
    log_name = "info_" + datetime.now().strftime("%m-%d_%H:%M") + ".log"
    logger = get_logger(log_dir, __name__, log_name, level=log_level)
    
    # Save the configuration file in the log directory
    with open(os.path.join(log_dir, data_name + "config2.yaml"), "w+") as _f:
        yaml.safe_dump(cfg, _f)
    
    logger.info(cfg)
    logger.info(dataset_path)
    logger.info(log_path)
    
    # multithreaded training method
    torch.set_num_threads(3)
    device = torch.device(cfg["device"])
    
    # Setting up the dataset
    dataloader = load_dataset(
        dataset_path,
        cfg["data"]["train_batch_size"],
        cfg["data"]["val_batch_size"],
        cfg["data"]["test_batch_size"],
        logger=logger,
    )

    # Setting up the model
    model_name = cfg["model_name"]
    input_dim = cfg["model"]["input_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    output_dim = cfg["model"]["output_dim"]
    num_nodes = cfg["data"]["cluster_num"]
    num_for_target = cfg["data"]["num_for_target"]
    num_for_predict = cfg["data"]["num_for_predict"]
    
    # Building diagram information
    if cfg["ablation"]["use_geo_affinity_matrix"]:
        geo_graph = np.load(os.path.join(base_path, "geo_affinity.npy")).astype(np.float32)
    else:
        geo_graph = np.eye(num_nodes, dtype=np.float32)
    
    if cfg["ablation"]["use_origin_destination_matrix"]:
        od_graph = np.load(os.path.join(base_path, "OD_affinity.npy")).astype(np.float32)
    else:
        od_graph = np.eye(num_nodes, dtype=np.float32)

    adjs = [geo_graph, od_graph]
    static_norm_adjs = [torch.tensor(asym_adj(adj)).to(device) for adj in adjs]

    print("geo_affinity", geo_graph)
    print("od_graph", od_graph)
    
    # Initialize the model based on user selection
    if name_model == "S":
        model = STNSCN(
            input_dim=cfg["model"]["input_dim"],
            time_dim=cfg["model"]["time_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            gcn_depth=cfg["model"]["gcn_depth"],
            alpha=cfg["model"]["alpha"],
            use_transform=cfg["model"]["use_transform"],
            fusion_mode=cfg["model"]["fusion_mode"],
            num_of_head=cfg["model"]["num_of_head"],
            dropout_prob=cfg["model"]["dropout_prob"],
            dropout_type=cfg["model"]["dropout_type"],
            device=device,
            num_of_weeks=cfg["data"]["num_of_weeks"],
            num_of_days=cfg["data"]["num_of_days"],
            num_of_hours=cfg["data"]["num_of_hours"],
            num_for_predict=cfg["data"]["num_for_predict"],
            num_for_target=cfg["data"]["num_for_target"],
            static_norm_adjs=static_norm_adjs,
            norm=cfg["model"]["dyn_norm"],
            use_curriculum_learning=cfg["train"]["use_curriculum_learning"],
            cl_decay_steps=cfg["train"]["cl_decay_steps"],
        )
    elif name_model == "G":
        model = DummyGRU(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            output_seq_len=num_for_predict,
        )
    elif name_model == "A":
        model = AGCRN(
            input_dim=cfg["model"]["input_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            output_seq_len=num_for_predict,
        )
    
    simple_model = 'dummy' in model_name.lower() or 'agcrn' in model_name.lower()
    # Multi-threaded training
    val_mae_list = []
    val_mape_list = []
    val_rmse_list = []
    mae_list = []
    mape_list = []
    rmse_list = []
    
    for i in range(cfg["runs"]):
        logger.info(model_name)
        if cfg["train"]["test_only"]:
            val_mae, val_mape, val_rmse, mae, mape, rmse = baseline_test(
                i, model, dataloader, device, logger, cfg
            )
        else:
            val_mae, val_mape, val_rmse, mae, mape, rmse = baseline_train(
                i,
                model,
                cfg["model_name"],
                dataloader,
                static_norm_adjs,
                device,
                logger,
                cfg,
                simple_model = simple_model
            )
        val_mae_list.append(val_mae)
        val_mape_list.append(val_mape)
        val_rmse_list.append(val_rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
    
    mae_list = np.array(mae_list)
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)
    amae = np.mean(mae_list, 0)
    amape = np.mean(mape_list, 0)
    armse = np.mean(rmse_list, 0)
    smae = np.std(mae_list, 0)
    smape = np.std(mape_list, 0)
    srmse = np.std(rmse_list, 0)
    
    logger.info("valid\tMAE\tRMSE\tMAPE")
    log = "mean:\t{:.4f}\t{:.4f}\t{:.4f}"
    logger.info(
        log.format(
            np.mean(val_mae_list), np.mean(val_rmse_list), np.mean(val_mape_list)
        )
    )
    log = "std:\t{:.4f}\t{:.4f}\t{:.4f}"
    logger.info(
        log.format(np.std(val_mae_list), np.std(val_rmse_list), np.std(val_mape_list))
    )
    logger.info("\n\n")
    logger.info("Test\tMAE\tRMSE\tMAPE")
    log = "mean:\t{:.4f}\t{:.4f}\t{:.4f}"
    logger.info(log.format(np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
    log = "std:\t{:.4f}\t{:.4f}\t{:.4f}"
    logger.info(log.format(np.std(mae_list), np.std(rmse_list), np.std(mape_list)))
    logger.info("\n\n")
import os
import shutil
import optuna
import logging

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmseg.utils import register_all_modules

# =========================================================
# 必须注册 mmseg 模块 + 配置日志
# =========================================================
register_all_modules()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_CFG = 'configs/lformer/lformer-b-YZ_seg-_256x256.py'
WORK_DIR = 'mmseg_log/lformer/l_eft_t_hpo_l'

# =========================================================
# 🔥 修复后的 Optuna Pruning + Metric Capture Hook
# =========================================================
class OptunaMetricHook(Hook):
    def __init__(self, trial=None, monitor='mIoU'):
        self.trial = trial
        self.monitor = monitor
        self.best_score = 0.0  # 记录最优分数

    def after_val_epoch(self, runner, metrics=None):
        """
        修复点：
        1. 增加metrics完整日志，方便调试
        2. 兼容不同的metrics结构（如无segm前缀的情况）
        3. 确保runner始终有final_val_score属性
        4. 记录最优分数而非最后一次分数
        """
        # 初始化属性，避免属性不存在
        if not hasattr(runner, 'final_val_score'):
            runner.final_val_score = 0.0

        # 打印完整metrics结构（关键调试信息）
        print_log(f"当前验证指标完整结构: {metrics}", logger=logger, level=logging.INFO)

        if metrics is None:
            print_log("警告：metrics为空，无法获取验证指标", logger=logger, level=logging.WARNING)
            return

        # 兼容两种常见的metrics结构（有/无segm前缀）
        score = None
        if 'segm' in metrics and self.monitor in metrics['segm']:
            score = metrics['segm'][self.monitor]
        elif self.monitor in metrics:
            score = metrics[self.monitor]
        else:
            print_log(f"警告：未找到监控指标 {self.monitor}，metrics键值: {list(metrics.keys())}", 
                     logger=logger, level=logging.WARNING)
            return

        # 只更新更优的分数
        if score > self.best_score:
            self.best_score = score
            runner.final_val_score = self.best_score

        print_log(f"当前trial {self.trial.number if self.trial else 'N/A'} - "
                 f"{self.monitor}: {score:.4f}, 最优: {self.best_score:.4f}", 
                 logger=logger, level=logging.INFO)

        # Optuna pruning（仅在有trial时执行）
        if self.trial is not None:
            self.trial.report(score, step=runner.epoch)  # 改用epoch更直观
            if self.trial.should_prune():
                print_log(f"Trial {self.trial.number} 触发pruning，当前分数: {score}", 
                         logger=logger, level=logging.INFO)
                raise optuna.TrialPruned()

# =========================================================
# 🎯 Optuna Objective（修复边界场景）
# =========================================================
def objective(trial):
    # ---------------- 搜索空间 ----------------
    embed_dim0 = trial.suggest_categorical('embed_dim0', [48, 64, 80])
    depth_stage3 = trial.suggest_int('depth_stage3', 20, 24)
    mlp_ratio = trial.suggest_float('mlp_ratio', 3.6, 4.8)

    lr = trial.suggest_float('lr', 3e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-3, log=True)

    embed_dims = [embed_dim0, embed_dim0*2, embed_dim0*4, embed_dim0*8]
    depths = [2, 2, depth_stage3, 2]
    mlp_ratios = [mlp_ratio] * 4

    # ---------------- 构建 cfg ----------------
    cfg = Config.fromfile(BASE_CFG)

    # Backbone
    cfg.model.backbone.embed_dims = embed_dims
    cfg.model.backbone.depths = depths
    cfg.model.backbone.mlp_ratios = mlp_ratios
    cfg.model.backbone.frozen_stages = 4

    # Decode head 对齐
    cfg.model.decode_head.in_channels = embed_dims

    # Optimizer
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.optim_wrapper.optimizer.weight_decay = weight_decay

    # HPO 专用短训练（修复：确保最后一步执行验证）
    cfg.train_cfg.max_iters = 10000
    cfg.train_cfg.val_interval = 2000  

    # Work dir
    cfg.work_dir = os.path.join(WORK_DIR, f'trial_{trial.number}')
    if os.path.exists(cfg.work_dir):
        shutil.rmtree(cfg.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)  # 确保目录存在

    # ---------------- Runner ----------------
    runner = Runner.from_cfg(cfg)

    # 🔥 修复：调整Hook优先级为NORMAL，确保在指标计算后执行
    runner.register_hook(
        OptunaMetricHook(trial, monitor='mIoU'),
        priority='NORMAL'  # 关键修改：从LOWEST改为NORMAL
    )

    # ---------------- 训练 ----------------
    try:
        runner.train()
    except optuna.TrialPruned:
        # Pruning是正常流程，返回当前最优分数
        print_log(f"Trial {trial.number} 被Prune，返回最优分数: {runner.final_val_score}", 
                 logger=logger, level=logging.INFO)
        return runner.final_val_score
    except Exception as e:
        print_log(f"Trial {trial.number} 训练失败: {str(e)}", logger=logger, level=logging.ERROR)
        return 0.0  # 失败时返回0分

    # ---------------- 返回最终 mIoU ----------------
    final_score = getattr(runner, 'final_val_score', 0.0)
    if final_score <= 0.0:
        print_log(f"Trial {trial.number} 未获取到有效分数，返回0", logger=logger, level=logging.WARNING)
    
    return final_score

# =========================================================
# 🚀 主入口
# =========================================================
if __name__ == '__main__':
    # 增加Optuna日志，方便调试
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction='maximize',
        study_name='l_EFT_T_hpo_1',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,       # 前5个trial不prune
            n_warmup_steps=5,         # 前5个epoch不prune（适配epoch维度）
            interval_steps=1          # 每1个epoch检查一次prune
        )
    )

    # 增加异常捕获，避免单个trial失败导致整个优化终止
    try:
        study.optimize(objective, n_trials=20, catch=(RuntimeError,))
    except Exception as e:
        print_log(f"超参优化过程出错: {str(e)}", logger=logger, level=logging.ERROR)

    print('=' * 60)
    print('Best mIoU:', study.best_value)
    print('Best Params:')
    for k, v in study.best_trial.params.items():
        print(f'{k}: {v}')

    # 保存最优参数到文件
    best_params_path = os.path.join(WORK_DIR, 'best_hpo_params.txt')
    with open(best_params_path, 'w') as f:
        f.write(f"Best mIoU: {study.best_value:.4f}\n")
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")
    print(f"\n最优参数已保存到: {best_params_path}")
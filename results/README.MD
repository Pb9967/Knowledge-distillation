**文件路径**：`Knowledge-distillation/results/README.md`

```markdown
# 实验结果目录 (Results Documentation)

本目录存储模型训练过程中生成的所有结果文件，包括模型检查点、训练历史、可视化图表和评估报告。


## 子目录结构

```bash
results/
├── teacher_performance/          # 教师模型评估结果（静态基准）
│   ├── visualizations/           # 可视化图表（混淆矩阵、箱线图等）
│   ├── detailed_results.csv      # 单张图像详细评估指标
│   └── evaluation_summary.json   # 评估摘要（JSON格式）
│
├── distillation_performance/     # 知识蒸馏训练结果（核心输出）
│   ├── checkpoint/               # 定期保存的检查点（每10轮）
│   │   ├── checkpoint_epoch10.pth
│   │   ├── checkpoint_epoch20.pth
│   │   └── ...
│   ├── best_model_ing/           # 训练过程中验证集最佳模型
│   │   ├── best_student_epoch72_iou0.4631.pth  # 格式示例
│   │   └── student_final_model.pth             # 最终保留的模型
│   ├── training_history.json     # 完整训练历史（损失、指标曲线数据）
│   ├── training_summary.txt      # 训练总结报告（文本格式）
│   └── distillation_training_history.png  # 训练曲线总览图
│
├── plot_history/                 # 自定义历史曲线图（可选）
└── inference/                    # 推理预测结果
    ├── inference_summary.json    # 批量推理统计结果
    └── *_result.png              # 单张图像可视化结果
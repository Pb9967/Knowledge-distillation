**文件路径**：`Knowledge-distillation/model/README.md`

```markdown
# 模型权重目录 (Model Weights Documentation)


## 目录结构

```bash
model/
├── teacher_deeplabv3p.pth          # 教师模型权重（由get_teacher.py生成）
├── student_final_model.pth         # 最终学生模型（蒸馏训练后生成，用于部署）
└── distillation_model/             # 中间模型文件（可选）
    └── （训练过程中保存的临时模型）
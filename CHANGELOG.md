# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范，版本遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增

- **三级姓名识别后端**：在线 PaddleOCR-VL API → 本地 PaddleOCR → 内置 `chineseocr_lite`，按优先级自动降级。
- **总分校验**：利用「总分 = Σ小题」约束自动修正总分，置信度置 0 提示人工核对。
- **异步识别 API**：`POST /api/recognize` 提交任务返回 `task_id`，`GET /api/result/<id>` 轮询结果，避免长耗时 OCR 阻塞服务。
- **日志系统**：全链路 `logging`，替代静默异常吞没，便于诊断降级/失败。
- **单元测试 + CI**：`pytest` 覆盖文本清洗与总分校验，GitHub Actions 自动运行。
- **Docker 部署**：`Dockerfile` + `docker-compose.yml` + `.dockerignore` 一键部署。
- **贡献规范**：`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、Issue/PR 模板。
- **在线 API 超时与预热**：`PADDLEOCR_API_TIMEOUT` 环境变量控制轮询上限，`warmup_paddleocr_api.py` 提前预热避免冷启动等待。

### 变更

- 推理参数（检测尺寸/置信度/NMS IoU/padding）抽为模块级常量，便于调整。
- README 补充演示动图、实测指标、架构图与完整目录结构。

### 修复

- 学号识别回退为**定长切分**：投影分割对含印刷横线的学号字段失效（10 位被合并为单个字符），实测确认后回退。

## 性能说明

以下为 val 集 72 张实测字符级准确率（真实手写数据）：

| 字段 | 字符级准确率 |
| --- | --- |
| 第2题(5) | 95.0% |
| 第2题(2) | 86.2% |
| 第2题(4) | 82.6% |
| 学号 | 56.4% |
| 总分（校验后） | 31.8% |

> 学号识别受手写粘连与印刷横线影响，为当前最弱环节；进一步提升需重新训练数字模型（详见 [README「当前状态与已知限制」](README.md#-当前状态与已知限制)）。

# 贡献指南

感谢你对 `answer-sheet-ocr` 的关注！欢迎通过 Issue、Pull Request 或讨论参与贡献。

## 快速上手

```bash
git clone https://github.com/Nothings2Seeyeyeye/answer-sheet-ocr.git
cd answer-sheet-ocr
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt                 # 安装测试依赖 pytest
```

## 运行测试

```bash
pytest -v
```

当前测试覆盖 `scripts/text_utils.py` 中的纯文本清洗函数（零第三方依赖）。请确保你的改动不破坏现有测试，并尽量为新逻辑补充测试。

## 代码风格

- 遵循 [PEP 8](https://peps.python.org/pep-0008/)，使用 4 空格缩进。
- 纯函数优先放入 `scripts/text_utils.py`，便于独立测试。
- 识别管线相关的类型标注使用 `from __future__ import annotations` 与 `typing`。
- 保持函数职责单一；对外接口需在 docstring 中说明。

## 提交信息规范

采用 Conventional Commits 风格：

- `feat:` 新功能
- `fix:` 修复
- `docs:` 文档
- `test:` 测试
- `refactor:` 重构
- `chore:` 构建/工具

示例：`feat: add PaddleOCR backend for name recognition`

## 提交 Issue

提交 Bug 或功能建议前，请先搜索已有 Issue 避免重复。请尽量提供：

- **Bug**：运行环境（OS/Python 版本）、复现步骤、期望与实际行为、相关日志。
- **功能建议**：使用场景、期望行为、可能的实现思路。

## 提交 Pull Request

1. Fork 本仓库并创建特性分支。
2. 保持改动聚焦，一个 PR 解决一个问题。
3. 通过 `pytest -v` 验证。
4. 在 PR 描述中说明改动动机与影响范围。

## 隐私约束（重要）

本项目明确**不收录**任何真实答题卡图片、真实姓名/学号明细、服务器凭据或 API key。请勿在 PR 中提交此类数据。

## 建议的贡献方向

- 打通/改进中文姓名识别（PaddleOCR 后端已预留接口）
- 提升总分与学号识别准确率（可加入 `总分 = Σ小题` 校验）
- 补充单元测试与 CI 覆盖
- 完善 Docker 部署与文档

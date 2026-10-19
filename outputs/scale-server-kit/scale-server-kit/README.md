# SCALE 服务器准备包

本包是复现准备工作，不包含未经核实的SCALE训练代码。完整方案见 PLAN.md。

## 上传与执行

在已经通过SSH连上学校服务器的VS Code窗口操作。截图显示目录名为 projects。

1. 将 scale-server-kit.zip 上传到服务器 ~/projects/（例如用SFTP上传；或先本地解压，再拖拽整个scale-server-kit目录到远程资源管理器的projects下）。
2. 如果上传的是zip，在服务器终端执行以下命令。若包已解压且目录存在，跳过解压，直接执行准备脚本。

```bash
cd "$HOME/projects"
python3 -c 'import pathlib,zipfile; p=pathlib.Path("scale-server-kit"); assert not p.exists(), "Folder already exists; skip extraction"; zipfile.ZipFile("scale-server-kit.zip").extractall(".")'
python3 scale-server-kit/prepare_scale.py --projects-root "$HOME/projects"
python3 scale-server-kit/prepare_scale.py --projects-root "$HOME/projects" --inventory-only
```

Python须>=3.9。无需pip/conda、无需服务器访问GitHub。准备脚本验证内置作者数据包SHA256，保留所有原始文件，不覆盖已存在的不同内容。

## 工作区

准备完成后，在远程VS Code中：文件 -> 从文件打开工作区 -> 选择
`~/projects/Projects-Only.code-workspace`。
也可在远程VS Code集成终端中执行：

```bash
code -r "$HOME/projects/Projects-Only.code-workspace"
```

若远程没有code命令，用上面的菜单。以后从“最近打开”选择这份工作区，不再打开整个用户主目录。
后续所有复现放在 `~/projects/refgrader-reproductions/<论文简称>/`，将自动显示。

## 交回结果以继续配置训练

```bash
cat "$HOME/projects/refgrader-reproductions/scale/reports/data_audit.json"
cat "$HOME/projects/refgrader-reproductions/scale/reports/"server_inventory_*.json
```

同时提供最终论文PDF及代码链接。正式训练命令会在模型、代码、协议与服务器GPU核实后给出。
重复准备不重下载、不重划分、不重训练；如果文件内容不一致，脚本停止并报告路径，不覆盖已有工作。

SCP可选（仅在你的本地电脑已配置对应SSH别名时）：
`scp scale-server-kit.zip LAB-SERVER:~/projects/`
本次助手环境无法解析LAB-SERVER，因此没有执行上传或任何服务器写入。

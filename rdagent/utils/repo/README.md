# RepoAnalyzer

RepoAnalyzer is a Python utility for analyzing and summarizing the contents of a Python repository. It provides a high-level overview of the repository structure, including a tree-like representation of the directory structure and details about files, classes, and functions.

## Features

- Generate a tree-like structure of the repository
- Summarize an entire repository
- Adjust verbosity levels for summaries
- Extract content from specific files
- Analyze Python files for classes and functions


## Usage

### Basic Usage

```python
from repo_utils import RepoAnalyzer

# Initialize the RepoAnalyzer with the path to your repository
repo_analyzer = RepoAnalyzer("/path/to/your/repo")

# Generate a summary of the repository
summary = repo_analyzer.summarize_repo()
print(summary)

# Extract content from specific files
highlighted_content = repo_analyzer.highlight(["file1.py", "file2.py"])
print(highlighted_content)
```

### Adjusting Verbosity Levels

You can adjust the verbosity of the summary using the following parameters:

- `verbose_level`: Controls the overall detail level of the summary
  - 0: Minimal (file names only)
  - 1: Default (file info, class names, function names)
  - 2+: Detailed (includes method details within classes)
- `doc_str_level`: Controls the inclusion of docstrings (0-2)
- `sign_level`: Controls the inclusion of function signatures (0-2)

Example:

```python
detailed_summary = repo_analyzer.summarize_repo(verbose_level=2, doc_str_level=1, sign_level=1)
print(detailed_summary)
```

## Example Output

### Repository Summary

```
Workspace Summary for my_project
========================================

Repository Structure:
my_project/
├── main.py
├── utils/
│   ├── helper.py
│   └── config.py
├── models/
│   ├── model_a.py
│   └── model_b.py

This workspace contains 5 Python files.

File 1 of 5:
File: main.py
----------------------------------------
This file contains 1 class and 2 top-level functions.

Class: MainApp
  Description: Main application class for the project.
  This class has 3 methods.

Function: setup_logging
  Accepts parameters: log_level
  Purpose: Configure the logging for the application.

Function: main
  Purpose: Entry point of the application.

...
```

### File Highlight

```python
highlighted_content = repo_analyzer.highlight(["main.py"])
print(highlighted_content["main.py"])
```

This will print the entire content of the `main.py` file.

## Key Components

### RepoAnalyzer Class

The main class that provides the functionality for analyzing repositories.

#### Methods:

- `summarize_repo(verbose_level=1, doc_str_level=1, sign_level=1)`: Generates a comprehensive summary of the repository, including a tree-like structure.
- `highlight(file_names)`: Extracts and returns the content of specified files.

### Tree-like Structure

The summary now includes a visual representation of the repository's directory structure, making it easier to understand the overall organization of the project.

---

# 中文说明

## RepoAnalyzer

RepoAnalyzer 是一个用于分析和总结 Python 代码仓库内容的 Python 工具。它能提供仓库结构的高级概览，包括目录结构的树状表示，以及关于文件、类和函数的详细信息。

## 功能特性

-   生成仓库的树状结构图
-   总结整个代码仓库
-   调整摘要的详细程度
-   提取特定文件的内容
-   分析 Python 文件中的类和函数

## 使用方法

### 基本用法

```python
from repo_utils import RepoAnalyzer

# 使用你的仓库路径初始化 RepoAnalyzer
repo_analyzer = RepoAnalyzer("/path/to/your/repo")

# 生成仓库的摘要
summary = repo_analyzer.summarize_repo()
print(summary)

# 提取特定文件的内容
highlighted_content = repo_analyzer.highlight(["file1.py", "file2.py"])
print(highlighted_content)
```

### 调整详细程度

你可以使用以下参数来调整摘要的详细程度：

-   `verbose_level`: 控制摘要的整体细节级别
    -   0: 最少信息（仅文件名）
    -   1: 默认（文件信息、类名、函数名）
    -   2+: 详细（包括类内部的方法细节）
-   `doc_str_level`: 控制是否包含文档字符串 (0-2)
-   `sign_level`: 控制是否包含函数签名 (0-2)

示例：

```python
detailed_summary = repo_analyzer.summarize_repo(verbose_level=2, doc_str_level=1, sign_level=1)
print(detailed_summary)
```

## 关键组件

### RepoAnalyzer 类

提供仓库分析功能的主类。

#### 方法：

-   `summarize_repo(verbose_level=1, doc_str_level=1, sign_level=1)`: 生成仓库的全面摘要，包括树状结构。
-   `highlight(file_names)`: 提取并返回指定文件的内容。

### 树状结构

摘要中现在包含仓库目录结构的可视化表示，使得理解项目的整体组织变得更加容易。

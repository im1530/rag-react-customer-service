"""
为整个工程项目提供统一的绝对路径
"""

import os


# 函数:获取工程项目所在的根目录
def get_project_root_path() -> str:
    """
    获取工程项目所在的根目录
    :return: 根目录的绝对路径(str)
    """
    # 当前文件的绝对路径(path_tool.py)
    current_file = os.path.abspath(__file__)

    # 先往上跳回一级,获取当前文件所在的绝对路径
    current_dir = os.path.dirname(current_file)

    # 再往上跳回一级,获取到真正的根目录
    project_root_path =  os.path.dirname(current_dir)

    return project_root_path



# 函数:获取传入的相对路径的绝对路径
def get_abs_path(relative_path) -> str:
    """
    传入相对路径,获取绝对路径
    :param relative_path:相对路径
    :return: 绝对路径(str)
    """
    # 先获取到整个工程项目的根目录
    project_root_path = get_project_root_path()

    # 将根目录和传入的相对路径拼接起来,获取绝对路径
    abs_path = os.path.join(project_root_path, relative_path)

    return abs_path



# 测试
if __name__ == '__main__':
    print(get_abs_path('data/test.txt'))


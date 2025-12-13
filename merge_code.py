import os

def generate_tree(dir_path, padding="", print_files=True, exclude_dirs=None, exclude_exts=None):
  """
  递归生成目录树字符串。
  """
  if exclude_dirs is None:
    exclude_dirs = []
  if exclude_exts is None:
    exclude_exts = []

  output_str = ""
  
  # 获取当前目录下所有条目并排序（文件夹在前，文件在后，或者按字母序）
  if not os.path.exists(dir_path):
    return "Directory not found."
    
  items = sorted(os.listdir(dir_path))
  
  # 过滤掉不需要的项目
  filtered_items = []
  for item in items:
    path = os.path.join(dir_path, item)
    if os.path.isdir(path):
      if item not in exclude_dirs:
        filtered_items.append(item)
    else:
      # 检查后缀
      if not any(item.endswith(ext) for ext in exclude_exts):
        filtered_items.append(item)

  # 遍历项目生成树结构
  count = len(filtered_items)
  for i, item in enumerate(filtered_items):
    path = os.path.join(dir_path, item)
    is_last = (i == count - 1)
    
    # 选择连接符
    connector = "└── " if is_last else "├── "
    
    output_str += padding + connector + item
    if os.path.isdir(path):
      output_str += "/\n"
      # 递归调用
      new_padding = padding + ("    " if is_last else "│   ")
      output_str += generate_tree(path, new_padding, print_files, exclude_dirs, exclude_exts)
    else:
      output_str += "\n"
      
  return output_str

def consolidate_project(source_dir, output_filename):
  """
  主函数：生成结构图 + 合并代码内容
  """
  # 定义排除列表
  exclude_dirs = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'build', 'train_log', 'tmp', 'trained_model', 'data', 'Test']
  exclude_extensions = ['.pyc', '.png', '.jpg', '.jpeg', '.gif', '.exe', '.bin', '.pkl', '.pth', 'igest', 'log']

  # 确保输出目录存在
  output_dir = os.path.dirname(output_filename)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  with open(output_filename, 'w', encoding='utf-8') as outfile:
    # 1. === 写入目录结构 (Directory Structure) ===
    root_name = os.path.basename(os.path.abspath(source_dir))
    outfile.write("Directory structure:\n")
    outfile.write(f"└── {root_name}/\n")
    
    # 生成树并写入
    tree_str = generate_tree(source_dir, padding="    ", exclude_dirs=exclude_dirs, exclude_exts=exclude_extensions)
    outfile.write(tree_str)
    outfile.write("\n\n") # 空几行分隔
    
    outfile.write("Files Content:\n")
    
    # 2. === 写入文件内容 (Files Content) ===
    for root, dirs, files in os.walk(source_dir):
      # 过滤目录
      dirs[:] = [d for d in dirs if d not in exclude_dirs]
      
      for filename in files:
        # 过滤文件后缀
        if any(filename.endswith(ext) for ext in exclude_extensions):
          continue
          
        file_path = os.path.join(root, filename)
        relative_path = os.path.relpath(file_path, source_dir)
        
        # 写入分隔符和文件名
        outfile.write("=" * 48 + "\n")
        outfile.write(f"FILE: {relative_path}\n")
        outfile.write("=" * 48 + "\n")
        
        try:
          with open(file_path, 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())
        except UnicodeDecodeError:
          outfile.write("[Binary file]\n")
        except Exception as e:
          outfile.write(f"[Error reading file: {e}]\n")
          
        outfile.write("\n\n")

  print(f"成功！目录树和代码已合并至: {output_filename}")

# ================= 运行脚本 =================
if __name__ == "__main__":
  # 设置目标路径
  target_directory = "." 
  output_file = "project_code.ingest"

  if os.path.exists(target_directory):
    consolidate_project(target_directory, output_file)
  else:
    print(f"错误: 找不到目录 '{target_directory}'")
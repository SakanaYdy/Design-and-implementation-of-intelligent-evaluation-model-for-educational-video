
import os
def get_video_files(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv'))]
    return video_files

def write_to_txt(video_files, output_file):
    with open(output_file, 'w') as f:
        for video_file in video_files:
            # 构造 写入文本的文本名
            filename = video_file.split('.')
            output = filename[0] + ".txt"
            command = f"""whisper {video_file} > {output}\n"""
            f.write(command)

def main():
    folder_path = input("请输入视频文件夹的路径：")
    output_file = input("请输入要保存的txt文件路径和名称：")

    video_files = get_video_files(folder_path)

    if not video_files:
        print("文件夹中没有找到视频文件。")
    else:
        write_to_txt(video_files, output_file)
        print(f"已将视频文件列表写入到 {output_file} 文件中。")

# C:\Users\杨大宇\PycharmProjects\pythonProject\Project\download
# C:\Users\杨大宇\PycharmProjects\pythonProject\Project\download\output
main()


# 读取运行命令到批处理文件
def copy_text_file(source_file, destination_file):
    try:
        with open(source_file, 'r') as source:
            content = source.read()
            with open(destination_file, 'w') as destination:
                destination.write(content)
        print(f"成功将 {source_file} 复制到 {destination_file}。")
    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
    except Exception as e:
        print(f"发生错误: {e}")

# C:\Users\杨大宇\PycharmProjects\pythonProject\Project\download\output
source_file = input("请输入源文本文件的路径：")
destination_file = input("请输入目标文本文件的路径：")

copy_text_file(source_file, destination_file)
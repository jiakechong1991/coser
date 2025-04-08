# -*- coding:utf-8 -*-
import os

"""列出指定目录下的txt文件，按照安装顺序进行合并，合并后保存到指定文件中"""

xiaoshuo_name = "fanren"
input_dir = "./data/my_data/source/{name}".format(name=xiaoshuo_name)
output_file = "data/my_data/source/{name}/{name}_merge.txt".format(name=xiaoshuo_name)

#如果输出文件存在，则删除
if os.path.exists(output_file):
    os.remove(output_file)


file_list = [item_file for item_file in os.listdir(input_dir) if item_file.endswith(".txt")]
file_list = sorted(file_list, key=lambda x: int(x.replace(".txt", "").split("_")[1]))
# print(file_list)
all_line_num = 0
break_num = 10
print("请确认，务必让文件列表，从前到后的进行合并，否则会乱序，请仔细检查文件列表！！！")


def test_exclue_line(this_line):
    """判断当前行是否包含排除的字符
    False： 包含
    True: 不包含
    """
    exclude_line = ["http", "小说", "手机"]
    flag = True
    for item_word in exclude_line:
        if item_word in this_line:
            flag = False
    
    return flag

def replace_word_line(this_line):
    replace_words = ["^^小说520 首 发^^", "**小说520 ***"]
    out_line = this_line
    for item_word in replace_words:
        if item_word in out_line:
            out_line = this_line.replace(item_word, "")
    return out_line

for item_file in file_list:

    print("     当前处理文件:{a}".format(a=item_file))
    # 读取文件
    this_file_lines = []
    with open(os.path.join(input_dir, item_file), "r", encoding="utf-8") as f:
        for item_line in f:
            item_line = replace_word_line(item_line)
            if test_exclue_line(item_line):  # 不包含 特定无效字符串
                this_file_lines.append(item_line)
                all_line_num += 1
            else:
                print(item_line.strip())
            
            if all_line_num > break_num:
                pass
                #1/0

    # 写入文件
    with open(output_file, "a", encoding="utf-8") as f_out:
        for content in this_file_lines:
            f_out.write(content)

print("输出文件:{a}".format(a=output_file))
print("这些被处理后的小说，可能还有特异的字符，还需要批量进行检查")
print("合并完成， 一共处理{a}个文件, 共计{b}行".format(a=len(file_list), b=all_line_num))












# -*- coding:utf-8 -*-
import os
import json


class CleanFormatStage(object):
    pass

    def __init__(self, lines:list):
        self.lines = lines
        self.src_lines_num = len(lines)
        self.this_stage_lines = lines  # 之后，每个管线阶段 处理这个
        self.next_stage_lines = [] 


    def all_ine_one(self):
        """ 整体的处理流程
        1. 判断 该行是否有问题
        2. 对问题进行修正
        """
        pass
        

        for item_line in self.this_stage_lines:
            
            if self.judege_line(this_line=item_line):  # 判定当前行 是否有问题
                # 没有问题
                self.next_stage_lines.append(item_line)
                continue

            else:  # 有问题，进行修复
                pass
                print(item_line)
            ### 修复流程

            # 1. 进行关键词替换
            item_line = self.replace_word_line(this_line=item_line)

            
                
            
            self.next_stage_lines.append(item_line)
        
        print("这些被处理后的小说，可能还有特异的字符，还需要批量进行检查")
        return  self.next_stage_lines
        
        

    def judge_by_rulebase(self, this_line):
        """判断当前行是否包含排除的字符
        False： 有异常
        True: 没有异常
        """
        flag = True
        exclude_line = ["http", "小说", "手机"]

        ####基于特定词语，进行识别判定
        for item_word in exclude_line:
            if item_word in this_line:
                flag = False
                break
        
        ####基于特征的正则模板，进行识别判定

        return flag
    
    def judege_line(self, this_line):
        """
        True: 没问题
        False： 有问题
        """
        flag = True
        if not self.judge_by_rulebase(this_line):  # 检查失败
            flag = False
        else:  # 这里可以加上一些基于迷你LLM或者其他小型model的方法，进行快速判定
            pass
        
        return flag
        


    def replace_word_line(self, this_line):
        """将 输入行 中的 特定字符串 替换掉"""
        # 当前的替换词表
        replace_words = ["^^小说520 首 发^^", "**小说520 ***"]
        out_line = this_line
        for item_word in replace_words:
            if item_word in out_line:
                out_line = this_line.replace(item_word, "")
        return out_line
    


class chapterSplist():
    """小说book必须支持chapter方式切分"""
    pass
    def __init__(self, book_lines):
        pass
        self.book_lines = book_lines
    








class MergeMulitFile(object):
    pass

    def __init__(self, input_dir, output_txt, output_json):
        self.all_lines = []
        self.input_dir = input_dir
        self.output_txt = output_txt
        self.output_json = output_json

    def merge_mulit_files(self):  
        # 将该目录下的book片段，有序进行合并 成一个完整book
        file_list = [item_file for item_file in os.listdir(self.input_dir) if item_file.endswith(".txt")]
        print(file_list)
        file_list = sorted(file_list, key=lambda x: int(x.replace(".txt", "").split("_")[1]))
        # print(file_list)
        print("请确认，务必让文件列表，从前到后的进行合并，否则会乱序，请仔细检查文件列表！！！")

        #如果输出文件存在，则删除
        if os.path.exists(self.output_txt):
            os.remove(self.output_txt)

        
        for item_file in file_list:

            print("     当前处理文件:{a}".format(a=item_file))
            # 读取文件
            with open(os.path.join(self.input_dir, item_file), "r", encoding="utf-8") as f:
                for item_line in f:
                    self.all_lines.append(item_line)
            

        # 写入文件
        with open(self.output_txt, "a", encoding="utf-8") as f_out:
            for content in self.all_lines:
                f_out.write(content)

        print("合并完成， 一共处理{a}个文件, 共计{b}行".format(a=len(file_list), b=len(self.all_lines)))
        print("输出文件:{a}".format(a=self.output_txt))
        return self.output_txt
    

    def merge_json(self):
        
        #如果输出文件存在，则删除
        if os.path.exists(self.output_json):
            os.remove(self.output_json)


        out_res = {
            "title": "凡人修仙传",
            "author": "忘语",
            "content": "".join(self.all_lines),
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(out_res, ensure_ascii=False, indent=4))
        print("已经合并成jsonl文件: {a}".format(a=output_json))
        return self.output_json



##### all_in_one流程
"""列出指定目录下的txt文件，按照安装顺序进行合并，合并后保存到指定文件中"""

xiaoshuo_name = "fanren"
input_dir = "./data/my_data/source/{name}".format(name=xiaoshuo_name)
output_file = "data/src/{name}_merge.txt".format(name=xiaoshuo_name)
output_json = "data/src/{name}_merge.jsonl".format(name=xiaoshuo_name)

merge_ins = MergeMulitFile(input_dir=input_dir, output_txt=output_file, output_json=output_json)
clean_format_ins = CleanFormatStage(merge_ins.all_lines)

merge_ins.merge_mulit_files()
merge_ins.all_lines = clean_format_ins.all_ine_one()
merge_ins.merge_json()







"""
说明：本代码 主要是清洗出一个良好的 小说版本

我们目标是构建一个pipeline， 可以自动化的校对小说，为后续流程提供一个【干净的输入】

目前 清洗和结构化 小说 存在如下问题：

###小说文本中 存在很多 插入性标识符:
1. 网站标识
2. 广告类标识
3. 新产品推广类标识
    1. 新小说即将退出，欢迎关注。。。。
4. 语气类标识：比如 
    1. 今天好累啊，明天再更新吧
    2. 未完待续

###按照章节切分困难：



####错别字等打字问题：
1. 本来是中文打字，结果按键问题，输入成了拼音
    妖魔：yaomo


"""










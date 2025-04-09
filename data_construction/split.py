
import re 
from typing import List, Tuple
from collections import Counter
import json
import jsonlines
from utils import cached

count_split_success = 0
count_split_failed = 0


def match_chapter_title(line_text):
    # 定义正则表达式
    pattern = r'(第[\u4e00-\u9fa5]+章)\s*([\u4e00-\u9fa5]+)'

    # 使用正则表达式进行匹配
    match = re.search(pattern, line_text)
    res = {
        "match_ok": False,
        "chapter_index": "",
        "chapter_title": ""

    }
    if match:
        chapter_index = match.group(1)  # 第一部分：章节序号
        chapter_title = match.group(2)    # 第二部分：标题
        res["chapter_index"] = chapter_index
        print("   章节识别   序号: {a}  标题:{b}".format(a=chapter_index, b=chapter_title))
        res["chapter_title"] = chapter_title
        res["src"] = line_text
        res["match_ok"] = True
    else:
        print("未匹配到内容")

    return res


def split_book(book: dict) -> list:
    """
    按照章节 进行book的切分
    
    Args:
    book (dict): The book dictionary containing 'content' key.
    
    Returns:
    dict: Updated book dictionary with 'content' as a list of chapter dictionaries.
    """
    content = book['content']
    
    #### 或者章节列表
    chapters_titles = []
    chapters_all = []

    one_temp_chapter_lines = []
    last_temp_chapter = None
    #begin_flag = False   # 出现 章节标题行，则开始记录

    for item_line in content.split('\n'):
        # 对该行进行标题匹配
        print("当前行：", item_line)
        temp_chapter = match_chapter_title(item_line.strip())
        if temp_chapter["match_ok"]:  # 匹配成功
            # 保存之前的章节识别结果
            if last_temp_chapter:
                chapters_all.append(
                    {
                        "chapter_src": last_temp_chapter["src"],  # 本文本行
                        "chapter_title": last_temp_chapter["chapter_title"],
                        "chapter_index": last_temp_chapter["chapter_index"],
                        "chapter_contents": one_temp_chapter_lines # list。 每个元素一行
                    })
            
            last_temp_chapter = temp_chapter
            one_temp_chapter_lines = []  # 先清空之前
            
                
        one_temp_chapter_lines.append(item_line)

    for i in chapters_all:
        print(i["chapter_src"])
    
    print("一共识别到{a}个章节".format(a=len(chapters_all)))
    print(chapters_all[3])
    1/0
    # 可以对 小/短 的章节进行合并

    return chapters_all





    # ####
    # # Process chapters given the boundaries
    # chapter_splits = []  #  [{"title": "Chapter 1", "content": "Chapter 1 content"}, ,,,]
    # for i, match in enumerate(chapters):
    #     start = match.start()
    #     end = chapters[i+1].start() if i+1 < len(chapters) else len(content)

    #     if i == 0:
    #         # add the content before the first chapter
    #         chapter_content = content[:match.start()].strip()
    #         chapter_title = None
    #         chapter_splits.append({"title": chapter_title, "content": chapter_content})

    #     chapter_title = match.group(0).strip()
    #     chapter_content = content[start:end].strip()
    #     chapter_splits.append({"title": chapter_title, "content": chapter_content})
    

    # return chapter_splits
    
    
    
if __name__ == '__main__':

    with jsonlines.open('data/src/books_example.jsonl', mode='r') as reader:
        books_data = list(reader) 

    # Process all books
    split_books = [split_book(book) for book in books_data]

    # Print the count 
    print(f"Split {count_split_success} books, failed {count_split_failed} books")
    # Update books_data with the processed books
    books_data = split_books

    print(f"Processed {len(books_data)} books, splitting their content into chapters.")

    
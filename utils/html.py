import os
from typing import List

import dominate
from dominate.tags import *


class HTML:
    """
    创建一个展示图片的网页
    title: 网页的名称
    web_dir: 保存网页 index.html 文件的文件夹
    img_dir: web_dir文件夹下一个存放需要展示图像的文件夹名
    """

    def __init__(self, title: str, img_dir: str, refresh: int = 0, file_name='index'):
        self.title = title
        self.img_dir = img_dir
        self.web_dir = os.path.dirname(self.img_dir)
        self.doc = dominate.document(title=title)
        self.file_name = file_name
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str: str):
        with self.doc:
            h3(str)

    def add_table(self, border: int = 1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, images: List[str], titles: List[str], width=512):
        """
        imgs: 图片的文件名
        txts: 图像展示的标题
        """
        self.add_table()
        with self.t:
            with tr():
                for image, title in zip(images, titles):
                    image_relative_path = os.path.join(
                        os.path.basename(self.img_dir), image
                    )
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        with p():
                            with a(href=image_relative_path):
                                img(
                                    style="width:%dpx" % (width),
                                    src=image_relative_path,
                                )
                            br()
                            p(title)

    def save(self):
        html_file = f"{self.web_dir}/{self.file_name}.html"
        f = open(html_file, "wt", encoding='utf-8')
        f.write(self.doc.render())
        f.close()

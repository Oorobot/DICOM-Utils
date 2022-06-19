import dominate
from dominate.tags import *
import os
import numpy as np


class HTML:
    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=512):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        with p():
                            with a(href=os.path.join("images", link)):
                                img(
                                    style="width:%dpx" % (width),
                                    src=os.path.join("images", im),
                                )
                            br()
                            p(txt)

    def save(self):
        html_file = "%s/index.html" % self.web_dir
        f = open(html_file, "wt")
        f.write(self.doc.render())
        f.close()


if __name__ == "__main__":

    web_dir = "results/ct2pet/test_50/"
    webpage = HTML(web_dir, "CT2PET")

    ims = []
    txts = []
    links = []

    for _, __, fnames in os.walk(os.path.join(web_dir, "images")):
        i = 0
        for fname in fnames:
            strs = fname.split("_")
            head3 = strs[0]
            tag = strs[1]
            ims.append(fname)
            txts.append(tag)
            links.append(fname)
    ims = np.array(ims).reshape((-1, 3))
    txts = np.array(txts).reshape((-1, 3))
    links = np.array(links).reshape((-1, 3))
    for i in range(0, ims.shape[0]):
        webpage.add_images(ims[i], txts[i], links[i], width=256)
    webpage.save()

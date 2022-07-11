import os
import dominate
from dominate.tags import *
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
    html = HTML(web_dir="Files/ThreePhaseBone/hip_mask", title="hip_mask")
    images = []
    txts = []
    links = []
    mask_images = []
    mask_links = []
    j = -1
    for _, __, fnames in os.walk(html.img_dir):
        for fname in fnames:
            j = (j + 1) % 26
            txt = os.path.basename(fname)
            txts.append(txt)
            if j == 25:
                mask_images.append(fname)
                mask_links.append(fname)
            else:
                images.append(fname)
                links.append(fname)
    j = 0
    for i in range(len(images) // 25):
        html.add_header(mask_images[i].split("_")[1])
        html.add_images(
            mask_images[i : i + 1], mask_images[:-4], mask_links[i : i + 1], 150
        )
        b = np.linspace(i * 25, i * 25 + 25, 6, dtype=np.uint8)
        for i in range(5):
            html.add_images(
                images[b[i] : b[i + 1]],
                txts[b[i] : b[i + 1]],
                links[b[i] : b[i + 1]],
                150,
            )
    html.save()

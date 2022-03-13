# -*- coding: utf-8 -*-
"""
@Project: 车载感知
@File   : io.py
@Author : Zhang P.H
@Date   : 2022/3/13
@Desc   :
"""
import json
import os.path


def write_json(file, data):
    if ".json" not in file:
        assert False, "json文件名不符合规范:".format(file)
    with open(file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def read_json(file):
    if ".json" not in file:
        assert False, "json文件名不符合规范"
    if not os.path.exists(file):
        assert False, "要读取的json文件不存在:{}".format(file)
    with open(file, "r", encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    pass

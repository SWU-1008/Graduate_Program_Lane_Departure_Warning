#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/3/24 14:51
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       python3 下载 google-drive 上的大文件
-------------------------------------------------
"""

__author__ = 'Max_Pengjb'

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    file_ids = [
        # '14FYFk4htz1Gx0J45z_lbeW2YRfu0nID_',
        # '1wK7gI1JdRVIcdjaSdkWHSRzk0OnF1r7E',
        '160zAWgFxbWGUAKMiI8tJcmSgSmBex5ZC',
        '11IZwS1Vu2zErlduJAEM_2M1sgRnRpUf2',
        '1je-h8YzAfn_aDiTcL4NAw8rKohvvMOcN',
        # '153n59sIFtkz270sTuhYsjXYjYeYgftQS',
        # '1bwXvlEX0RYkixR3LDHte-U_KpMMIRG8x',
        # '1fpoOPVxbQbFmmLd2Pe2u8LnYueZLGtOm'
    ]  # 自行更换
    destinations = [
        # '2011_09_30_drive_0028.bag',
        # 'campus_large_dataset.bag',
        'campus_small_dataset.bag',
        'garden_dataset.bag',
        'park_dataset.bag',
        # 'rooftop_ouster_dataset.bag',
        # 'rotation_dataset.bag',
        # 'walking_dataset.bag'
    ]  # 自定义
    for file_id, destination in zip(file_ids, destinations):
        download_file_from_google_drive(file_id, destination)

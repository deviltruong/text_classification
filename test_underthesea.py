# -*- coding: utf-8 -*-
from underthesea import word_tokenize

sentence = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'
token = word_tokenize(sentence, format="text")
print(token)
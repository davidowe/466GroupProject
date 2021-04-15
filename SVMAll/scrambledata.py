from io import open
import csv
import random
lang_list = ["BG","CS","DA","DE","EL","EN","ES","ET","FI","FR","HR","HU","IT","LT","LV","LV","MT","NL","PL","PT","RO","SK","SL","SV"]
with open('testing_VERY_FEW.tsv','w',encoding="utf8") as f:
    for lang in lang_list:
        lang_file_list = list(filter(None,[x[:-3] for x in open("data/" + lang + ".tsv", 'r',encoding='utf-8').read().split("\n")]))
        sentences = lang_file_list[-1000:]
        output = csv.writer(f,delimiter='\t')
        for sentence in sentences:
            output.writerow([sentence,lang])

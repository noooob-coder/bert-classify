import xlrd
import random
from tqdm import tqdm
import pandas as pd
class Process(object):
    def __init__(self):
        self.base_data_path='./data/data.xlsx'
        self.sheet_name = u'' #填入sheet名
        self.text_name=u'' #填入文本列名
        self.label_name=u'' #填入标签类名
    def data_split(self,full_list, a1, a2, a3, shuffle=False):  # 对数据按比例分类
        n_total = len(full_list)
        offset1 = int(n_total * a1 / 10)
        offset2 = int(n_total * (a1 + a2) / 10)
        if n_total == 0 or offset1 < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset1]
        sublist_2 = full_list[offset1:offset2]
        sublist_3 = full_list[offset2:]
        return sublist_1, sublist_2, sublist_3
    def to_text(self,data_list, name, path, item=False):  # 将数据写入文本
        file = open(path, 'w', encoding='utf-8')
        file.truncate()
        print("正在写入%s数据" % (name))
        if item == True:
            for i in tqdm(data_list):
                file.write(i + "\n")
            file.close()
        else:
            if name == False:
                for j in tqdm(data_list):
                    for n in j:
                        n = n.replace(" ", "").replace("\n", '').replace("\r", '')
                        file.write(n + '-----')
                    file.write('\n')
                file.close()
            else:
                random.shuffle(data_list)
                for i in tqdm(data_list):
                    i[0] = i[0].replace(" ", "").replace("\n", '').replace("\r", '')
                    file.write(i[0] + '-----' + i[1])
                    file.write('\n')
                file.close()
        print("写入完成")
    def data_load(self):
        df=pd.read_excel(self.base_data_path,sheet_name=self.sheet_name)[[self.text_name,self.label_name]].dropna()
        data_list=df.values.tolist()
        return data_list
    def count(self):
        class_list = []
        data_list = self.data_load()
        data_dic = {}
        test = []
        train = []
        dev = []
        for i in data_list:
            data_dic.setdefault(i[1], []).append(i[0])
        for key in data_dic:
            num = len(data_dic[key])
            # print(key+" "+str(num))
            if num >= 1000:  # 总数大于1000的计入数据
                class_list.append(key)
        # print(out_list)
        self.to_text(class_list, name='class', path='./data/class.txt', item=True)
        for i in class_list:
            data_class = []
            for j in data_dic[i]:
                data_class.append([j, i])
            a, b, c = self.data_split(data_class, 7, 2, 1, shuffle=True)
            train = train + a
            test = test + b
            dev = dev + c
        print("共有训练数据%d条，测试数据%d条，验证数据%d条" % (len(train), len(test), len(dev)))
        self.to_text(train, name="train", path='./data/train.txt')
        self.to_text(test, name="test", path='./data/test.txt')
        self.to_text(dev, name="dev", path='./data/dev.txt')
        for i in class_list:
            sum_class = 0
            for m in data_list:
                if m[1] == i:
                    sum_class += 1
            print("%s类有%d条" % (i, sum_class))
if __name__ == '__main__':
    p=Process()
    p.count()
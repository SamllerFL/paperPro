import random


def make_corpus(type):
    with open('../file/corpus.txt', 'a', encoding='utf-8') as f:
        if type == 'testTowork':
            rd = random.randint(100, 9000)
            f.write('将压变'+str(rd)+"#刀闸手车由“试验”位置摇至“工作”位置\n")
            f.write('检查压变' + str(rd) + "#刀闸手车确在“工作”位置\n")
            f.write('合上压变' + str(rd) + "#刀闸\n")
            f.write('检查压变' + str(rd) + "#刀闸三相确已合上\n")
        elif type == 'workTotest':
            rd = random.randint(100, 9000)
            f.write('拉开压变' + str(rd) + "#开关\n")
            f.write('检查压变' + str(rd) + "#开关三相确在分闸位置\n")
            f.write('将压变'+str(rd)+"#刀闸手车由“工作”位置摇至“试验”位置\n")
            f.write('检查压变' + str(rd) + "#刀闸手车确在“试验”位置\n")
        else:
            raise Exception("The type is not a right one.")


def make_corpus2(type):
    with open('../file/corpus2.txt', 'a', encoding='utf-8') as f:
        if type == 'testTowork':
            rd = ''
            f.write('将压变'+str(rd)+"刀闸手车由“试验”位置摇至“工作”位置\n")
            f.write('检查压变' + str(rd) + "刀闸手车确在“工作”位置\n")
            f.write('合上压变' + str(rd) + "刀闸\n")
            f.write('检查压变' + str(rd) + "刀闸三相确已合上\n")
        elif type == 'workTotest':
            rd = ''
            f.write('拉开压变' + str(rd) + "开关\n")
            f.write('检查压变' + str(rd) + "开关三相确在分闸位置\n")
            f.write('将压变'+str(rd)+"刀闸手车由“工作”位置摇至“试验”位置\n")
            f.write('检查压变' + str(rd) + "刀闸手车确在“试验”位置\n")
        else:
            raise Exception("The type is not a right one.")


def make_corpus3(type):
    with open('../file/corpus3.txt', 'a', encoding='utf-8') as f:
        if type == 'testTowork':
            rd = random.randint(100, 3000)
            f.write('将压变' + str(rd) + "#刀闸手车由“试验”位置摇至“工作”位置\n")
            f.write('检查压变' + str(rd) + "#刀闸手车确在“工作”位置\n")
            f.write('合上压变' + str(rd) + "#刀闸\n")
            f.write('检查压变' + str(rd) + "#刀闸三相确已合上\n")

            rd = random.randint(3000, 6000)
            f.write('将万悦一段' + str(rd) + "#开关手车由“试验”位置摇至“工作”位置\n")
            f.write('检查万悦一段' + str(rd) + "#开关手车确在“工作”位置\n")
            f.write('合上万悦一段' + str(rd) + "#开关\n")
            f.write('检查万悦一段' + str(rd) + "#开关三相确已合上\n")

            rd = random.randint(6000, 9000)
            f.write('将母联' + str(rd) + "#开关手车由“试验”位置摇至“工作”位置\n")
            f.write('检查母联' + str(rd) + "#开关手车确在“工作”位置\n")
            f.write('合上母联' + str(rd) + "#开关\n")
            f.write('检查母联' + str(rd) + "#开关三相确已合上\n")

        elif type == 'workTotest':
            rd = random.randint(100, 3000)
            f.write('拉开压变' + str(rd) + "#开关\n")
            f.write('检查压变' + str(rd) + "#开关三相确在分闸位置\n")
            f.write('将压变'+str(rd)+"#刀闸手车由“工作”位置摇至“试验”位置\n")
            f.write('检查压变' + str(rd) + "#刀闸手车确在“试验”位置\n")

            rd = random.randint(3000, 6000)
            f.write('拉开万悦一段' + str(rd) + "#开关\n")
            f.write('检查万悦一段' + str(rd) + "#开关手车确在分闸位置\n")
            f.write('将万悦一段'+str(rd)+"#开关手车由“工作”位置摇至“试验”位置\n")
            f.write('检查万悦一段' + str(rd) + "#开关手车确在“试验”位置\n")

            rd = random.randint(6000, 9000)
            f.write('拉开母联' + str(rd) + "#开关\n")
            f.write('检查母联' + str(rd) + "#开关手车确在分闸位置\n")
            f.write('将母联' + str(rd) + "#开关手车由“工作”位置摇至“试验”位置\n")
            f.write('检查母联' + str(rd) + "#开关手车确在“试验”位置\n")

        else:
            raise Exception("The type is not a right one.")


n = 1500
for i in range(n):
    if i % 2:
        make_corpus3('testTowork')
    else:
        make_corpus3('workTotest')


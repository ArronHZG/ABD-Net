with open("train_list.txt", "r+") as f:
    lines = f.readlines()
label_image = {}
for line in lines:
    name, label = line.split(" ")
    label = label[:-1]
    # print(name," ",label)
    if label in label_image.keys():
        label_image[label].append(name)
    else:
        label_image[label] = [name]

# print(label_image.keys())
print(len(label_image.keys()))

numbers = dict()

with open("id_number.txt",'w+') as f:
    for k, v in label_image.items():
        f.write(f"{k} {len(v)}\n")
        if len(v) in numbers.keys():
            numbers[len(v)] += 1
        else:
            numbers[len(v)] = 1

sorted_keys = sorted(numbers)


with open("number_statistic.txt",'w+') as f:
    for key in sorted_keys:
        f.write(f'{key} {numbers[key]}\n')

# with open("query_list.txt", "w+") as qf:
#     with open("gallery_list.txt", "w+") as gf:
#         for k, v in label_image.items():
#             if len(v) > 2:
#                 qf.write(f"{v[0]} {k}\n")
#                 for i in range(1, len(v)):
#                     gf.write(f"{v[i]} {k}\n")
#             else:
#                 for i in range(len(v)):
#                     gf.write(f"{v[i]} {k}\n")

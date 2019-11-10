with open("train_list.txt","r+") as f:
    lines = f.readlines()
label_image = {}
for line in lines:
    name,label = line.split(" ")
    label=label[:-1]
    print(name," ",label)
    if label in label_image.keys():
        label_image[label].append(name)
    else:
        label_image[label] = [name]

print(label_image.keys())
print(len(label_image.keys()))

with open("query_list.txt","w+") as qf:
    with open("gallery_list.txt","w+") as gf:
        for k, v in label_image.items():
            qf.write(f"{v[0]} {k}\n")
            for i in range(1,len(v)):
                gf.write(f"{v[i]} {k}\n")



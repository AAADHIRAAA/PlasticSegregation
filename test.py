import os
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']
train_dir = 'samples/train'
validation_dir = 'samples/test'
for class_name in class_names:
    print(f"{class_name}: {len(os.listdir(os.path.join(train_dir, class_name)))} images")

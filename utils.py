import os
import torchvision.transforms as transforms

# directory = the path of dir which includes all images
# train_split_file = dog_1 and cat_1 and so on

def find_all_files_over(directory,train_split_file):
    
    path_list = []
    with open(train_split_file) as f:
        data = f.readlines()
        
        for line in data:
            line = line.strip().split(" ")[0]
            # line = line.partition('\n')[0]
            
            file_path = os.path.join(directory,line)    
            files =os.listdir(file_path)
            files.sort()
            # if len(files) > 10:
            #     print(line,files)
            # print(len(files))
            # print(line,files,len(files))
            for file in files:
                    
                    yield os.path.join(file_path, file)


def find_all_files_label(directory,train_split_file):
    
    path_list = []
    with open(train_split_file) as f:
        data = f.readlines()
        for line in data:
            ele = line.strip().split(" ")
            line = ele[0]
            label = int(ele[1])
            
            # line = line.partition(' ')[0]
            
            file_path = os.path.join(directory,line)    
            files =os.listdir(file_path)
            files.sort()
            # if len(files) > 10:
            #     print(line,files)
            # print(len(files))
            # print(line,files,len(files))
            for file in files:
                    
                    yield label

def norm(img, vgg=False):
    if vgg:
        # normalize for pre-trained vgg model
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize [-1, 1]
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    return transform(img)
import os
import shutil

# # def delite_dir():
#     delite_dir_path = '../dogs-vs-cats/datasets'
#     dir_name = ["dog", "cat"] 
#     for i in range(11):
#         for append_name in dir_name:
#             delite_dir_path =  '../dogs-vs-cats/datasets/' + append_name + "_" + str(i)  
#             os.rmdir(delite_dir_path)

# # dir作成
# def make_dir():
#     new_dir_path = '../dogs-vs-cats/datasets'
#     dir_name = ["dog", "cat"]
#     for i in range(11):
#         for append_name in dir_name: 
#             new_dir_path = '../dogs-vs-cats/datasets/' + append_name + "_" + str(i)
#             os.mkdir(new_dir_path)

# move file
def move_file():
    files_path = "../dogs-vs-cats/datasets/escape_cat"
    dog_dir = "../dogs-vs-cats/datasets/dog/"
    cat_dir = "../dogs-vs-cats/datasets/cat/"
    files =os.listdir(files_path)

    dog_count = 1
    cat_count = 1

    for file_path in files:
        if "dog" in file_path:
            print("dog")
            file_num = (dog_count-1) // 1250
            file_num += 1
            dog_dir = "../dogs-vs-cats/datasets/dog_" + str(file_num)
            print(dog_dir)
            dog_count += 1
            new_path = shutil.move(files_path + "/" + file_path, dog_dir)

        elif "cat" in file_path:
            print("cat")
            file_num = (cat_count-1) // 1250
            file_num += 1
            cat_dir = "../dogs-vs-cats/datasets/cat_" + str(file_num)
            # cat_dir = "../dogs-vs-cats/datasets/escape_cat"

            cat_count += 1 
            print(cat_dir)
            new_path = shutil.move(files_path + "/" + file_path, cat_dir)
    
    print("dog", dog_count-1)
    print("cat", cat_count-1)


# move_file()

import csv
import glob
import sys
from csv import writer
import shutil
import os
import random
import torch

shutil.rmtree('/home/turingsleftfoot/machine_learning/code/myViT variable tokens/torch_batches/', ignore_errors=False, onerror=None)
os.mkdir('/home/turingsleftfoot/machine_learning/code/myViT variable tokens/torch_batches/')
os.mkdir('/home/turingsleftfoot/machine_learning/code/myViT variable tokens/torch_batches/testing')
os.mkdir('/home/turingsleftfoot/machine_learning/code/myViT variable tokens/torch_batches/training')

shutil.rmtree('/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/batches/', ignore_errors=False, onerror=None)

src_dir = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/tokens/'
dest_dir = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/batches/tokens/'
files = os.listdir(src_dir)
shutil.copytree(src_dir, dest_dir)

src_dir = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/patches/'
dest_dir = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/batches/patches/'
files = os.listdir(src_dir)
shutil.copytree(src_dir, dest_dir)



my_classes = ['chainsaw', 'church', 'dog', 'fish', 'french_horn', 'golf_ball', 'media_player', 'parachute', 'petrol_pump', 'rubbish_truck']
root_path = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/batches/'
the_largest_number = 0
the_largest_file = 'hello'
token_dimension = 0
global_batch_paddingQ = False
randomiseQ = True
batch_size = 4
#running on my laptop need to use batch of 4 max :'(

my_patches_files = []
my_training_patches_files = []
my_testing_patches_files = []

for i in range(len(my_classes)):
    for j in range(2):
        
        if j == 0: t_t = '/test/'
        if j == 1: t_t = '/train/'
        
        patches_input_files = root_path + 'patches' + t_t + my_classes[i] +'/*'
        new_files = glob.glob(patches_input_files)

        if j == 0:
            my_testing_patches_files.append(new_files)
        else:
            my_training_patches_files.append(new_files)
            
        my_patches_files.append(new_files)

my_testing_patches_files = [item for sublist in my_testing_patches_files for item in sublist]
my_training_patches_files = [item for sublist in my_training_patches_files for item in sublist]
my_patches_files = [item for sublist in my_patches_files for item in sublist]
my_patches_files.sort()
my_testing_patches_files.sort()
my_training_patches_files.sort()


if global_batch_paddingQ == True:
    for k in range(len(my_patches_files)):
        fp = open(my_patches_files[k],'r')
        candidate = len(fp.readlines())
        if the_largest_number < candidate: 
            the_largest_number = candidate
            the_largest_file = my_patches_files[k]
        fp.close()



if randomiseQ == True:
    random.shuffle(my_training_patches_files)
if len(my_training_patches_files) % batch_size == 0:
    num_of_training_batches = int(len(my_training_patches_files)/batch_size)
else:
    num_of_training_batches = int(len(my_training_patches_files)/batch_size)+1

if len(my_testing_patches_files) % batch_size == 0:
    num_of_testing_batches = int(len(my_testing_patches_files)/batch_size)
else:
    num_of_testing_batches = int(len(my_testing_patches_files)/batch_size)+1


batched_training_patches_files = ['']*num_of_training_batches
batched_testing_patches_files = ['']*num_of_testing_batches
for i in range(num_of_testing_batches):
    if len(my_testing_patches_files) % batch_size == 0:
        batched_testing_patches_files[i] = my_testing_patches_files[i*batch_size:(i+1)*batch_size]
        
    else:
        if i < num_of_testing_batches - 1:
            batched_testing_patches_files[i] = my_testing_patches_files[i*batch_size:(i+1)*batch_size]
        else:
            remainder_size = len(my_testing_patches_files) % batch_size
            batched_testing_patches_files[i] = my_testing_patches_files[i*batch_size:i*batch_size+remainder_size]
            
            
for i in range(num_of_training_batches):
    if len(my_training_patches_files) % batch_size == 0:
        batched_training_patches_files[i] = my_training_patches_files[i*batch_size:(i+1)*batch_size]
        
    else:
        if i < num_of_training_batches - 1:
            batched_training_patches_files[i] = my_training_patches_files[i*batch_size:(i+1)*batch_size]
        else:
            remainder_size = len(my_training_patches_files) % batch_size
            batched_training_patches_files[i] = my_training_patches_files[i*batch_size:i*batch_size+remainder_size]


tokens_per_training_batch = [0] * num_of_training_batches
tokens_per_testing_batch = [0] * num_of_testing_batches
largest_file_in_training_batch = [''] * num_of_training_batches
largest_file_in_testing_batch = [''] * num_of_testing_batches

if global_batch_paddingQ == False:
    for i in range(len(batched_testing_patches_files)):
        for k in range(len(batched_testing_patches_files[i])):
            fp = open(batched_testing_patches_files[i][k],'r')
            candidate = len(fp.readlines())
            if tokens_per_testing_batch[i] < candidate: 
                tokens_per_testing_batch[i] = candidate
                largest_file_in_testing_batch[i] = batched_testing_patches_files[i][k]
            fp.close()
    
    for i in range(len(batched_training_patches_files)):
        for k in range(len(batched_training_patches_files[i])):
            fp = open(batched_training_patches_files[i][k],'r')
            candidate = len(fp.readlines())
            if tokens_per_training_batch[i] < candidate: 
                tokens_per_training_batch[i] = candidate
                largest_file_in_training_batch[i] = batched_training_patches_files[i][k]
            fp.close()
    


tokens_input_file = root_path + 'tokens/test/chainsaw/my_tokens0.csv'
fp = open(tokens_input_file,'r')
first_line = fp.readline()
token_dimensions = first_line.count(',') + 1 
fp.close()

batched_testing_tokens_files = batched_testing_patches_files.copy()
batched_training_tokens_files = batched_training_patches_files.copy()
for i in range(len(batched_testing_tokens_files)):
    batched_testing_tokens_files[i] = [sub.replace('patches', 'tokens') for sub in batched_testing_patches_files[i]]
for i in range(len(batched_training_tokens_files)):   
    batched_training_tokens_files[i] = [sub.replace('patches', 'tokens') for sub in batched_training_patches_files[i]]


to_add_to_patch_files = [-1]
to_add_to_token_files = [0]*token_dimensions

for i in range(len(batched_testing_tokens_files)):           
    for k in range(len(batched_testing_tokens_files[i])):
        
        fp = open(batched_testing_patches_files[i][k],'r')
        num_of_lines = len(fp.readlines())
        fp.close()
        
        if global_batch_paddingQ == True:
            chosen_number = the_largest_number
        else:
            chosen_number = tokens_per_testing_batch[i]
            
        if num_of_lines < chosen_number:
            with open(batched_testing_patches_files[i][k], 'a') as fp:     
                num_of_zeros_to_add = chosen_number - num_of_lines
                writer_object = writer(fp)
                while num_of_zeros_to_add != 0:
                    writer_object.writerow(to_add_to_patch_files)
                    num_of_zeros_to_add = num_of_zeros_to_add - 1
                fp.close()
    
            with open(batched_testing_tokens_files[i][k], 'a') as fp:     
                num_of_zeros_to_add = chosen_number - num_of_lines
                writer_object = writer(fp)
                while num_of_zeros_to_add != 0:
                    writer_object.writerow(to_add_to_token_files)
                    num_of_zeros_to_add = num_of_zeros_to_add - 1
                fp.close()


for i in range(len(batched_training_tokens_files)):           
    for k in range(len(batched_training_tokens_files[i])):
        
        fp = open(batched_training_patches_files[i][k],'r')
        num_of_lines = len(fp.readlines())
        fp.close()
        
        if global_batch_paddingQ == True:
            chosen_number = the_largest_number
        else:
            chosen_number = tokens_per_training_batch[i]
            
        if num_of_lines < chosen_number:
            with open(batched_training_patches_files[i][k], 'a') as fp:     
                num_of_zeros_to_add = chosen_number - num_of_lines
                writer_object = writer(fp)
                while num_of_zeros_to_add != 0:
                    writer_object.writerow(to_add_to_patch_files)
                    num_of_zeros_to_add = num_of_zeros_to_add - 1
                fp.close()
    
            with open(batched_training_tokens_files[i][k], 'a') as fp:     
                num_of_zeros_to_add = chosen_number - num_of_lines
                writer_object = writer(fp)
                while num_of_zeros_to_add != 0:
                    writer_object.writerow(to_add_to_token_files)
                    num_of_zeros_to_add = num_of_zeros_to_add - 1
                fp.close()


real_classes = []
for i in range(len(my_classes)):
    path = '/home/turingsleftfoot/machine_learning/code/image_to_token_for_imagenette/batches/patches/test/' + my_classes[i]
    if len(os.listdir(path)) > 0:
        real_classes.append(my_classes[i])
        

torch_batches_path = '/home/turingsleftfoot/machine_learning/code/myViT variable tokens/torch_batches/'
 
for i in range(len(batched_testing_tokens_files)):
    first_time_lucky = True
    for j in range(len(batched_testing_tokens_files[i])):

        with open(batched_testing_tokens_files[i][j], 'r') as f: 
            csv_reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC) 
            my_tokens = list(csv_reader)
        f.close()
        
        my_tokens = torch.FloatTensor(my_tokens)
        my_tokens = torch.unsqueeze(my_tokens, 0)
        
        if first_time_lucky == True:
            my_input_x = my_tokens
        else:
            my_input_x = torch.cat((my_input_x, my_tokens), 0)
        
        #use commented lines instead of non commented if you want to do one hot encoding
        #truth = torch.zeros(len(real_classes)).long()
        truth = torch.zeros(1).long()
        for k in range(len(real_classes)):
            if real_classes[k] in batched_testing_tokens_files[i][j]:        
                #truth[k] = 1
                truth[0] = k
                break
        #truth = torch.unsqueeze(truth, 0)
        
        if first_time_lucky == True:
            my_input_y = truth
            first_time_lucky = False
        else:
            my_input_y = torch.cat((my_input_y,truth), 0)

    torch.save(my_input_x, torch_batches_path + 'testing/testing_x' + str(i) + '.pt')
    torch.save(my_input_y, torch_batches_path + 'testing/testing_y' + str(i) + '.pt')
    
 
for i in range(len(batched_training_tokens_files)):
    first_time_lucky = True
    for j in range(len(batched_training_tokens_files[i])):

        with open(batched_training_tokens_files[i][j], 'r') as f: 
            csv_reader = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC) 
            my_tokens = list(csv_reader)
        f.close()
        
        my_tokens = torch.FloatTensor(my_tokens)
        my_tokens = torch.unsqueeze(my_tokens, 0)
        
        if first_time_lucky == True:
            my_input_x = my_tokens
            
        else:
            my_input_x = torch.cat((my_input_x, my_tokens), 0)
        
        #use commented lines instead of non commented if you want to do one hot encoding
        #truth = torch.zeros(len(real_classes)).long()
        truth = torch.zeros(1).long()
        for k in range(len(real_classes)):
            if real_classes[k] in batched_training_tokens_files[i][j]:        
                #truth[k] = 1
                truth[0] = k
                break
        #truth = torch.unsqueeze(truth, 0)
        
        if first_time_lucky == True:
            my_input_y = truth
            first_time_lucky = False
        else:
            my_input_y = torch.cat((my_input_y,truth), 0)
    
    torch.save(my_input_x, torch_batches_path + 'training/training_x' + str(i) + '.pt')
    torch.save(my_input_y, torch_batches_path + 'training/training_y' + str(i) + '.pt')
     
        
        
        
        
        
        
        
        
        
#

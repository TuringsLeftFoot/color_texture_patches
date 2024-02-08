import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import csv
import scipy
import statistics
import glob

#start_time = time.time()

patches_output_path = '/home/turingsleftfoot/machine_learning/'
tokens_output_path = '/home/turingsleftfoot/machine_learning/'
image_path = '/home/turingsleftfoot/machine_learning/mountain.jpg'

### sample here means color texture sample

image = Image.open(image_path)
data = np.array(image)
my_image = data/256.0
if my_image.ndim == 2: 
    my_image = np.tile(my_image, (3, 1, 1))
    my_image = np.swapaxes(my_image, 0, 1)
    my_image = np.swapaxes(my_image, 1, 2)

w_pre, h_pre = len(my_image[0]), len(my_image)
sample_size = int(((w_pre+h_pre)/2.0)*3.25/200)
if sample_size < 2:
    sample_size = 2

w = w_pre-(w_pre % sample_size)
h = h_pre-(h_pre % sample_size)
num_samples_w = int(w/sample_size)
num_samples_h = int(h/sample_size)
num_samples = int(num_samples_w * num_samples_h)
image_to_be_patched = my_image[:h,:w]
 
samples_rgb = np.zeros(shape=(num_samples, sample_size, sample_size, 3)) 
samples_red = np.zeros(shape=(num_samples, sample_size, sample_size))
samples_green = np.zeros(shape=(num_samples, sample_size, sample_size))
samples_blue = np.zeros(shape=(num_samples, sample_size, sample_size))
sample_vectors_red = np.zeros(shape=(num_samples, sample_size * sample_size))
sample_vectors_green = np.zeros(shape=(num_samples, sample_size * sample_size))
sample_vectors_blue = np.zeros(shape=(num_samples, sample_size * sample_size))
raw_sample_vectors_red = np.zeros(shape=(num_samples, sample_size * sample_size))
raw_sample_vectors_green = np.zeros(shape=(num_samples, sample_size * sample_size))
raw_sample_vectors_blue = np.zeros(shape=(num_samples, sample_size * sample_size))
sample_vector_length_red = np.zeros(shape=(num_samples))
sample_vector_length_green = np.zeros(shape=(num_samples))
sample_vector_length_blue = np.zeros(shape=(num_samples))
red_length_comp = np.zeros(shape=(num_samples,num_samples))
green_length_comp = np.zeros(shape=(num_samples,num_samples))
blue_length_comp = np.zeros(shape=(num_samples,num_samples))
similarity_image = np.zeros(shape=(num_samples_h, num_samples_w,3))
similarity_score_add = np.zeros(shape=(num_samples_h, num_samples_w))
similarity_score_mul = np.zeros(shape=(num_samples_h, num_samples_w))

for p in range(num_samples):

    samples_rgb[p] = image_to_be_patched[(p % num_samples_h) * sample_size : ((p % num_samples_h)+1) * sample_size, int(p / num_samples_h) * sample_size : (int(p / num_samples_h)+1) * sample_size] 
    samples_red[p] = image_to_be_patched[(p % num_samples_h) * sample_size : ((p % num_samples_h)+1) * sample_size, int(p / num_samples_h) * sample_size : (int(p / num_samples_h)+1) * sample_size,0] 
    samples_green[p] = image_to_be_patched[(p % num_samples_h) * sample_size : ((p % num_samples_h)+1) * sample_size, int(p / num_samples_h) * sample_size : (int(p / num_samples_h)+1) * sample_size,1]
    samples_blue[p] = image_to_be_patched[(p % num_samples_h) * sample_size : ((p % num_samples_h)+1) * sample_size, int(p / num_samples_h) * sample_size : (int(p / num_samples_h)+1) * sample_size,2] 
    
    sample_vectors_red[p] = np.reshape(samples_red[p], sample_size * sample_size, order='F') 
    sample_vectors_green[p] = np.reshape(samples_green[p], sample_size * sample_size, order='F') 
    sample_vectors_blue[p] = np.reshape(samples_blue[p], sample_size * sample_size, order='F') 
    
    sample_vectors_red[p] = np.sort(sample_vectors_red[p])
    sample_vectors_green[p] = np.sort(sample_vectors_green[p])
    sample_vectors_blue[p] = np.sort(sample_vectors_blue[p])
    
    raw_sample_vectors_red[p] = sample_vectors_red[p]
    raw_sample_vectors_green[p] = sample_vectors_green[p]
    raw_sample_vectors_blue[p] = sample_vectors_blue[p]
    
    sample_vector_length_red[p] = np.linalg.norm(sample_vectors_red[p])
    sample_vector_length_green[p] = np.linalg.norm(sample_vectors_green[p])
    sample_vector_length_blue[p] = np.linalg.norm(sample_vectors_blue[p])
    
    if sample_vector_length_red[p] == 0: sample_vector_length_red[p] = 0.0000001
    if sample_vector_length_green[p] == 0: sample_vector_length_green[p] = 0.0000001
    if sample_vector_length_blue[p] == 0: sample_vector_length_blue[p] = 0.0000001

    sample_vectors_red[p] = sample_vectors_red[p] / sample_vector_length_red[p]
    sample_vectors_green[p] = sample_vectors_green[p] / sample_vector_length_green[p]
    sample_vectors_blue[p] = sample_vectors_blue[p] / sample_vector_length_blue[p]

sample_similarity_red = sample_vectors_red @ sample_vectors_red.T
sample_similarity_green = sample_vectors_green @ sample_vectors_green.T
sample_similarity_blue = sample_vectors_blue @ sample_vectors_blue.T


    
red_length_comp = np.outer(sample_vector_length_red,1/sample_vector_length_red)
red_length_comp = np.where(red_length_comp > 1, 1/red_length_comp, red_length_comp)

green_length_comp = np.outer(sample_vector_length_green,1/sample_vector_length_green)
green_length_comp = np.where(green_length_comp > 1, 1/green_length_comp, green_length_comp)

blue_length_comp = np.outer(sample_vector_length_blue,1/sample_vector_length_blue)
blue_length_comp = np.where(blue_length_comp > 1, 1/blue_length_comp, blue_length_comp)

 
sample_similarity_red = np.multiply(sample_similarity_red,red_length_comp)
sample_similarity_green = np.multiply(sample_similarity_green,green_length_comp)
sample_similarity_blue = np.multiply(sample_similarity_blue,blue_length_comp)

# sample_similarity_red = (sample_similarity_red + 1) / 2
# sample_similarity_green = (sample_similarity_green + 1) / 2
# sample_similarity_blue = (sample_similarity_blue + 1) / 2

sample_similarity_red = np.where(sample_similarity_red > 1, 1, sample_similarity_red)
sample_similarity_green = np.where(sample_similarity_green > 1, 1, sample_similarity_green)
sample_similarity_blue = np.where(sample_similarity_blue > 1, 1, sample_similarity_blue)
            
print("Sampling image complete")
    

mini_patch_threshold = 0.8

def p_to_yx(p,num_samples_h):
    
    return int(p%num_samples_h), int(p/num_samples_h)

def yx_to_p(y,x,num_samples_h):
    
    return int(x*num_samples_h + y)

def test_surroundings(similarity_score_mul, in_patch, tested, num_samples_h,num_samples_w,mini_patch_threshold):
    
    to_test = np.setdiff1d(in_patch, tested)
        
    for i in range(len(to_test)):
        y_zero,x_zero = p_to_yx(to_test[i], num_samples_h)
        for j in range(-1,2):
            for k in range(-1,2):
                if (j == 0 and k == 0) == False:
                    if (y_zero == 0 and k == -1) == False and (y_zero == num_samples_h-1 and k == 1) == False and (x_zero == 0 and j == -1) == False and (x_zero == num_samples_w-1 and j == 1) == False:         
                        p = to_test[i]+j*num_samples_h+k
                        y,x = p_to_yx(p, num_samples_h)
                        if similarity_score_mul[y,x] > mini_patch_threshold and (p in in_patch) == False:
                            in_patch = np.concatenate((in_patch, np.array([p])), axis=0)

    
    if tested[0] == -1:
        tested = to_test
    else:
        tested = np.concatenate((tested, to_test), axis=None)

    return in_patch, tested

def remove_offcuts(candidate_patch, in_patch, tested, num_samples_h, num_samples_w):
    
    to_test = np.setdiff1d(in_patch, tested)
        
    for i in range(len(to_test)):
        for j in range(-1,2):
            for k in range(-1,2):
                if (j == 0 and k == 0) == False:
                    p = to_test[i]+j*num_samples_h+k
                    if (p in candidate_patch) == True and (p in in_patch) == False:
                        in_patch = np.concatenate((in_patch, np.array([p])), axis=0)
    
    if tested[0] == -1:
        tested = to_test
    else:
        tested = np.concatenate((tested, to_test), axis=None)
        
    return in_patch, tested




candidate_patches = []
candidate_patches_nums = []
print("Finding candidate patches")

for test_sample_num in range(num_samples):
    if test_sample_num % int(num_samples/5) == 0 and test_sample_num > 0:
        print(str(int(100.0*test_sample_num/num_samples)+1), "%")
        
    in_patch = np.array([test_sample_num])
    tested = np.array([-1])
    
    for p in range(num_samples):
        y,x = p_to_yx(p, num_samples_h)
        similarity_score_mul[y][x] = sample_similarity_red[test_sample_num][p] * sample_similarity_green[test_sample_num][p] * sample_similarity_blue[test_sample_num][p]
        
    done_at_least_once = False
    while len(in_patch)>len(tested) or done_at_least_once == False:
        done_at_least_once = True
    
        in_patch, tested = test_surroundings(similarity_score_mul, in_patch, tested, num_samples_h, num_samples_w, mini_patch_threshold)
    
    candidate_patches.append(in_patch.tolist())
    candidate_patches_nums.append(len(in_patch)) 



my_patches = []
print("Finding best patches")
do_now = False
while sum(len(v) for v in my_patches) < num_samples:
    # if sum(len(v) for v in my_patches) > 0:
    #     print(str(int(sum(len(v) for v in my_patches)*100.0/num_patches)), " %")
    
    biggest_index = candidate_patches_nums.index(max(candidate_patches_nums))
    
    my_patches.append(candidate_patches[biggest_index])
    
    to_deduct = candidate_patches.pop(biggest_index)
    
    del candidate_patches_nums[biggest_index]
    
   
    for i in range(len(to_deduct)):
        for j in reversed(range(len(candidate_patches))):
            if to_deduct[i] == candidate_patches[j][0]:
                del candidate_patches[j]
                del candidate_patches_nums[j]
                
    for i in range(len(candidate_patches)):
        both = set(to_deduct).intersection(candidate_patches[i])
        if len(both) > 0:
            candidate_patches_nums[i] = candidate_patches_nums[i] - len(both)
            candidate_patches[i] = [x for x in candidate_patches[i] if x not in both]
            
    
    #remove offcuts
    for test_patch_num in range(len(candidate_patches)):
        in_patch = np.array([candidate_patches[test_patch_num][0]])
        tested = np.array([-1])
        
        done_at_least_once = False
        while len(in_patch)>len(tested) or done_at_least_once == False:
            done_at_least_once = True
            
            in_patch, tested = remove_offcuts(candidate_patches[test_patch_num], in_patch, tested, num_samples_h, num_samples_w)
        
        candidate_patches[test_patch_num] = in_patch.tolist()
        candidate_patches_nums[test_patch_num] = len(in_patch)
                
with open(patches_output_path + "my_patches.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(my_patches)


print("Total number of patches used:", str(len(my_patches)))
#print(str(int(np.sum(similarity_score_mul)*100.0/num_patches)), "% of image covered by", str(how_many_patches), "biggest patches") 



token_length = 100
num_sample_points = token_length - 19
num_outer_sample_points = int(num_sample_points/2)
my_outer_samples = [None]*len(my_patches)
my_internal_samples = [None]*len(my_patches)

#find all internal and outer patches
for x in range(len(my_patches)):
    my_outer_samples[x] = []
    my_internal_samples[x] = []
    
    for i in range(len(my_patches[x])):
        we_are_done = False
        for j in range(-1,2):
            for k in range(-1,2):
                if (j == 0 and k == 0) == False and we_are_done == False:
                    p = my_patches[x][i]+j*num_samples_h+k
                    if (p in my_patches[x]) == False:
                        my_outer_samples[x].append(my_patches[x][i])
                        we_are_done = True
        if we_are_done == False:
            my_internal_samples[x].append(my_patches[x][i])

#sample outer patches
my_patch_samples = [None]*len(my_patches)
to_tick_off = [None]*len(my_patches)
for x in range(len(my_outer_samples)):
    my_patch_samples[x] = []
    to_tick_off[x] = my_outer_samples[x]
    outer_interval = int(len(my_outer_samples[x])/num_outer_sample_points)
    if outer_interval == 0:
        outer_interval = 1
    counter = 0
    #other_counter = 0
    escape_options = [to_tick_off[x][0]]
    #while len(to_tick_off[x]) > 0 and other_counter < num_outer_sample_points:
    while len(to_tick_off[x]) > 0:
        if len(escape_options) == 0:
            escape_options = [to_tick_off[x][0]]
        for j in range(-1,2):
            for k in range(-1,2):
                if (j == 0 and k == 0) == False:
                    p = escape_options[0]+j*num_samples_h+k
                    if (p in to_tick_off[x]) == True and (p in escape_options) == False:
                        escape_options.insert(1, p)
        counter = counter + 1
        if counter % outer_interval == 0:
            my_patch_samples[x].append(escape_options[0])
            #other_counter = other_counter + 1
        my_index = to_tick_off[x].index(escape_options[0])
        del to_tick_off[x][my_index]
        del escape_options[0]
        
         

#sample internal patches
for x in range(len(my_internal_samples)):
    num_internal_sample_points = num_sample_points - len(my_patch_samples[x])
    to_tick_off[x] = my_internal_samples[x]
    other_counter = 0

    for i in range(len(my_internal_samples[x])):
        to_delete = [my_internal_samples[x][i]]
        counter = 0
        if other_counter < num_internal_sample_points:
            for j in range(-1,2):
                for k in range(-1,2):
                    if (j == 0 and k == 0) == False:
                        p = my_internal_samples[x][i]+j*num_samples_h+k
                        if (p in to_tick_off[x]) == True:
                            counter = counter + 1
                            to_delete.append(p)
        if counter == 8:
            my_patch_samples[x].append(my_internal_samples[x][i])
            other_counter = other_counter + 1
            to_tick_off[x] = [m for m in to_tick_off[x] if m not in to_delete]


my_red = [[]]*len(my_patches)
my_green = [[]]*len(my_patches)
my_blue = [[]]*len(my_patches)
my_tokens = [[]]*len(my_patches)

print("")
print("making tokens")

def add_stats(to_return, my_list):
    
    to_return = to_return + [statistics.mean(my_list)]
    to_return = to_return + [statistics.median(my_list)]
    to_return = to_return + [scipy.stats.mode(my_list, keepdims = False)[0]]
    to_return = to_return + [scipy.stats.tstd(my_list)]
    if np.isnan(scipy.stats.skew(my_list)) == True:
        to_return = to_return + [0.0]
    else:
        to_return = to_return + [scipy.stats.skew(my_list)]
        
    if np.isnan(scipy.stats.kurtosis(my_list, fisher=True)) == True:
        to_return = to_return + [0.0]
    else:
        to_return = to_return + [scipy.stats.kurtosis(my_list, fisher=True)]
    
    return to_return

def add_sample_positions(to_return, my_list):
    
    to_return = to_return + [statistics.mean(my_list)]
    
    return to_return

for x in range(len(my_patches)):
    for p in range(len(my_patches[x])):
        my_red[x] = my_red[x] + raw_sample_vectors_red[my_patches[x][p]].tolist()
        my_green[x] = my_green[x] + raw_sample_vectors_green[my_patches[x][p]].tolist()
        my_blue[x] = my_blue[x] + raw_sample_vectors_blue[my_patches[x][p]].tolist()
    
    my_tokens[x] = add_stats(my_tokens[x], my_red[x])
    my_tokens[x] = add_stats(my_tokens[x], my_green[x])
    my_tokens[x] = add_stats(my_tokens[x], my_blue[x])
    my_tokens[x] = my_tokens[x] + [num_samples_h / num_samples]
    my_tokens[x] = my_tokens[x] + [(element+1) / num_samples for element in my_patch_samples[x]]
    num_pads = token_length - len(my_tokens[x])
    if num_pads > 0:
        padding = [0] * num_pads
        my_tokens[x] = my_tokens[x] + padding

with open(tokens_output_path + "my_tokens.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(my_tokens)

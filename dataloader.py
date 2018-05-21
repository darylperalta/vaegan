'''Dataloader for VAE/GAN'''


import glob
import os
import numpy as np
import cv2

def dataloader(path='/home/daryl/datasets/img_align_celeba', image_size=64,batch_size =64, normalized = True):
    """Generator to be used with model.fit_generator()."""
    '''
    print(os.path.join(path,'*.jpg'))
    files = glob.glob(os.path.join(path,'*.jpg'))
    files.sort()
    print(len(files))
    print(files[0:10])
    print(files[-2:])
    '''
    while True:
        image_list = glob.glob(os.path.join(path,'*.jpg'))
        #print('len_image_list', len(image_list))
        #print('batch_size', batch_size)
        num_batches = int(len(image_list)/batch_size)
        remainder = len(image_list)%batch_size
        #print('remainder', remainder)
        #print('num batch', num_batches)
        np.random.shuffle(image_list)
        del(image_list[-remainder:])
        #print(len(image_list))
        '''split list of image file names'''
        image_list_split = np.array_split(image_list,num_batches)
        #print('length of image list split',len(image_list_split))
        #print(image_list_split[0][0:10])
        #print('new epoch')
        while image_list_split:

            batch_image_list = image_list_split.pop()
            #print('len batch image list',len(batch_image_list) )
            batch_images = np.zeros((len(batch_image_list),image_size,image_size,3),dtype=np.float32)
            for i in range(len(batch_image_list)):
                img_temp = cv2.imread(batch_image_list[i])
                #cv2.imshow('temp',img_temp)
                #cv2.waitKey(0)
                batch_images[i,:,:,:] = cv2.resize(img_temp, (image_size,image_size))


            if normalized:
                batch_images = batch_images/255.0
                #cv2.imshow('sample',batch_images[i,:,:,:])
                #cv2.waitKey(0)
            yield batch_images, None

def main():
    some_gen = dataloader()
    #a,b = next(some_gen)
    #a,b = next(some_gen)
    #print('a', type(a))
    while(True):

        a,b = next(some_gen)
        print('a', type(a))
        print(a.shape)
    #dataloader()
if __name__ == '__main__':
    main()

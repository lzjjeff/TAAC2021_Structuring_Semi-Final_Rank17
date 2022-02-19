import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.train:
        print("Replace bad videos of train set...")
        train_video_without_same_region_dir = '/home/tione/notebook/dataset/videos/train_without_same_region/'
        train_video_dir = '/home/tione/notebook/dataset/videos/video_5k/train_5k/'

        with open('./pre/train_5k_bad.txt', 'r') as f:
            bad_filenames = f.read()
    
        bad_filenames = bad_filenames.split('\n')
    
        for filename in bad_filenames:
            train_video_filename = os.path.join(train_video_dir, filename + '.mp4')
            command = 'cp {} {}'.format(train_video_filename, train_video_without_same_region_dir)        
            os.system(command)

        print("Finished.")
        
    if args.inference:
        print("Replace bad videos of test set...")
        test_video_dir = '/home/tione/notebook/dataset/videos/test_5k_2nd/'
        test_video_without_same_region_dir = '/home/tione/notebook/dataset/videos/test_2nd_without_same_region/'
    
        with open('./pre/test_5k_2nd_bad.txt', 'r') as f:
            bad_filenames = f.read()
    
        bad_filenames = bad_filenames.split('\n')
    
        for filename in bad_filenames:
            test_video_filename = os.path.join(test_video_dir, filename + '.mp4')
            command = 'cp {} {}'.format(test_video_filename, test_video_without_same_region_dir)        
            os.system(command)
            
        print("Finished.")


    

import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import os 
import random
import torch
from torchvision import transforms as T
from PIL import Image, ImageFile
# read a image through PIL and transfer it to tensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("using load_truncated_images, to avoid bug")
#import pdb;pdb.set_trace()


class full_head_clean_real_data_temporal(Dataset):
    def __init__(self, root_path, image_transform=None, sample_frame=8, flag='real', more_image_control = True):
        # super()
        self.root = root_path
        self.flag = flag
        self.potential_face_list = [ '10', '12', '13','14', '15', '16' ,'17','18', '19','20', '21', '22', '23', '24', '25', '26', 
                                    '27','28', '29', '30', '31','32','33', '34' ,'35','36' '37','38', '54', '55', '56','57'
                                      ]
        self.potential_face_list_PH = ['00', '01', '02', '03', '04', '05','06','07','08','09','10','11',
                                       '25','26','27','28','29', '30', '31', '32', '33', '34', '35']
        self.potential_NS_fine_face_list = ['16','18','19','25','26','28','31','55','56' ]
        self.potential_NS_fine_back_list = ['59','50','49','48','46','45','01','00','02']
        self.potential_PH_fine_face_list = [str(i) for i in range(0,9)]  + [str(j) for j in range(63,71)]
        #self.potential_PH_fine_face_list([])##['00','01','02','03','04','05','06','32','33','34','35' ]
        self.potential_PH_fine_back_list = [str(i) for i in range(29,46)]#['16','17','18','19','20','21']
        self.id_list = []
        self.more_image_control = more_image_control
        self.id_driving_list_PH = []
        # for id_name in sorted(os.listdir(self.root)):
        #     if id_name.startswith('PH_'):
        #         camera_folder = os.path.join(self.root, id_name, 'condition')
        #     else:
        #         camera_folder = os.path.join(self.root, id_name, 'camera')
        #     if not os.path.isdir(os.path.join(self.root, id_name, 'image')):
        #         continue
        #     # if image folder doenst have image, skip
        #     elif not any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for file in os.listdir(camera_folder)):
        #         continue
        #     elif id_name.startswith('seed'): # remove the PanoHead stupid shit !
        #         continue
        #     else:
        #         self.id_list.append(id_name)
        for id_name in sorted(os.listdir(self.root)):
            if id_name.startswith('PHsup_'): # remove the
                self.id_driving_list_PH.append(id_name)  
            if id_name.startswith('PHsup_'): #or id_name.startswith('0') or id_name.startswith('i'): # remove the
            #if id_name.startswith('seed'):
                # if id_name.startswith("i"):
                #     # check the subfolder if it had a subfolder called images
                #     image_folder = os.path.join(self.root, id_name,'images')
                #     if not any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for file in os.listdir(image_folder)):
                #         print("not have a png folder in:", image_folder)
                #         continue
                    # elif int(id_name.split("_")[1])>1000:
                    #     continue 
                    # else:
                    #    self.id_list.append(id_name)
                #else:
                    self.id_list.append(id_name)
            else:
                continue
        self.transform = image_transform
        self.sample_frame = sample_frame
    def __len__(self):
        
        return 100000
    
    def __getitem__(self, idx):
        if self.sample_frame == 1:
            raise KeyError("sample_frame should be at least larger than 1")
            
        len_id_list = len(self.id_list)
        #print('len_id_list totall:', len_id_list)
        idx = random.randint(0, len_id_list-1)
        extra_appearance_list = []
        id_name = self.id_list[idx]
        #
        len_id_drive_PH = len(self.id_driving_list_PH)
        idx_drive = random.randint(0, len_id_drive_PH-1)
        id_drive_name_PH = self.id_driving_list_PH[idx_drive]    
        if id_name.startswith('PHsup_'):
            gt_path_ = os.path.join(self.root, id_name, "image")
            valid_face_list, valid_all_list = self.check_face(gt_path_, 'PH', potential_face_list = self.potential_PH_fine_face_list)
            valid_no_face_list, _ = self.check_face(gt_path_, 'PH', potential_face_list =self.potential_PH_fine_back_list)
            condition_image_path_ = os.path.join(self.root, id_drive_name_PH, "image")
            
        elif id_name.startswith('0'): # for nersemble dataset
            gt_path_ = os.path.join(self.root, id_name, "image_seg")
            valid_face_list, valid_all_list = self.check_face(gt_path_, 'real', potential_face_list =self.potential_NS_fine_face_list)
            valid_no_face_list, _ = self.check_face(gt_path_, 'real', potential_face_list =self.potential_NS_fine_back_list)
            condition_image_path_ = os.path.join(self.root, id_name, "camera")
            
        elif id_name.startswith('i'): # stylization dataset
            valid_face_list = ['0.png']
            valid_all_list = ['0.png','1.png','2.png','3.png']
            condition_dict = {'0.png':'0000000.png','1.png':'0000008.png','2.png':'0000018.png','3.png':'0000027.png'}
            valid_no_face_list = ['2.png']
            gt_path_ = os.path.join(self.root, id_name, "images")
            condition_image_path_ = os.path.join(self.root, id_drive_name_PH, "condition")
        random.shuffle(valid_all_list)
        random.shuffle(valid_face_list)
        random.shuffle(valid_no_face_list)
        # appearance frame idx should be randomly picked from 0-9 27-35 
        if id_name.startswith('i'):
            appearance_frame_idx = '0.png'
            if self.more_image_control:
                more_appearance_frame_idx = '2.png'
            frames_list = valid_all_list
        else:
            valid_exlcude_face_list = list(set(valid_all_list) - set(valid_face_list))
            random.shuffle(valid_exlcude_face_list)
            appearance_frame_idx = valid_face_list[0].zfill(7)+'.png'
            frames_list = valid_all_list[:self.sample_frame]
            #print(id_name, " ")
            #print(valid_face_list[0])
            if self.more_image_control:
                if len(valid_no_face_list) == 0:
                    more_appearance_frame_idx = valid_exlcude_face_list[-1].zfill(7)+'.png'
                    print('idName:', id_name, 'appearance:', appearance_frame_idx, ', doesnt have no back image')
                more_appearance_frame_idx = valid_no_face_list[0].zfill(7)+'.png'
        # gt _path_
        
        apperance  = self.read_image(os.path.join(gt_path_, appearance_frame_idx))
        targets = []
        conditions = []
        interval = random.randint(2, 6)
        frames_list = self.pick_frames(num_frames=self.sample_frame, total_frames=72, interval=interval)
        #
        print('frame_list:', frames_list[0:4], 'interval:', interval)
        for frame in frames_list:
            if not id_name.startswith('i'):
                target_frame_idx = str(frame).zfill(7)+'.png' 
                #source_path = camera_folders[1]
                condition = self.read_image(os.path.join(condition_image_path_, target_frame_idx))         
            else:
                target_frame_idx = frame  
                condition = self.read_image(os.path.join(condition_image_path_, condition_dict[target_frame_idx]))         
            target = self.read_image(os.path.join(gt_path_, target_frame_idx))            
   
            # mask condition here
            if self.transform is not None:
                target = self.transform(target)  
                condition = self.transform(condition)
            #condition = np.ones_like(target).astype(np.uint8)
            targets.append(target)
            conditions.append(condition)
        targets = torch.stack(targets)
        conditions = torch.stack(conditions)
        prompt = ''
        if self.transform is not None: 
            condition_image = self.transform(apperance)      
        
        # mask condition here
        
        #condition = np.ones_like(target).astype(np.uint8) 
        
        res = {'image': targets, 'condition_image': condition_image, 'condition': conditions,'text_bg':prompt, 'text_blip': prompt}
        if self.more_image_control:
            more_appearance_frame = self.read_image(os.path.join(gt_path_, more_appearance_frame_idx))
            if self.transform is not None:
                more_appearance_frame = self.transform(more_appearance_frame) # we only consider
            #extra_appearance_list.append(more_appearance_frame)
            res['extra_appearance'] = more_appearance_frame        
        return res
    def read_image(self, path):
        #print("using_path:", path)
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np
    def check_face(self, id_path, flags, potential_face_list=None):  
        valid_face = []
        for face_id in potential_face_list:
            image_path = os.path.join(id_path, face_id.zfill(7)+'.png')
            if os.path.isfile(image_path):
                valid_face.append(face_id.zfill(7))
        valid_all = []
        for face_id in os.listdir(id_path):
            if face_id.endswith('.png'):
                valid_all.append(face_id.split('.')[0])
        return valid_face, valid_all  
    def pick_frames(self, num_frames, total_frames=72, interval=None):
        # 如果没有指定间隔，则默认为1（即连续帧）
        if interval is None:
            interval = 1
        
        # 随机选择一个起始帧
        start_frame = random.randint(0, total_frames - 1)
        
        # 计算连续四帧
        selected_frames = [(start_frame + i * interval) % total_frames for i in range(num_frames)]
        
        return selected_frames









class back_head_generation(Dataset):
    def __init__(self, inference_image_dataset, condition_path, image_transform=None):
        self.root = inference_image_dataset
        self.condition_path = condition_path
        self.id_list = []
        for id_name in sorted(os.listdir(self.root)):
            if id_name.endswith('.png') or id_name.endswith('.jpg'):
                self.id_list.append(id_name)
        self.transform = image_transform
        
    def __len__(self):
        
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id_name = self.id_list[idx]
        #target_frame_idx = str().zfill(7)+'.png'
        #gt_path_ = os.path.join(self.root, id_name)
        condition_image_path_ = os.path.join(self.condition_path, "0000018.png")
        condition = self.read_image(condition_image_path_)
        #condition = self.read_image(os.path.join(self.condition_path, id_name))#self.read_image(condition_image_path_)
        appearance_path = os.path.join(self.root, id_name)
        appearance = self.read_image(appearance_path)

        prompt = ''
        if self.transform is not None:
            #target = self.transform(target)  
            condition_image = self.transform(appearance)
            condition = self.transform(condition)        
        fea_condition = condition_image
        res = {'image': condition_image, 'condition_image': condition_image, 'condition': condition,'text_bg':prompt, 'text_blip': prompt, "image_name": id_name, 'fea_condition': fea_condition}
        
        return res
    def read_image(self, path):
        #print("using_path:", path)
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np
    

class full_head_clean_inference_final_face(Dataset):
    def __init__(self, condition_path, image_transform=None, inference_image_dataset= None, initial_image_path = None, train=False, style_front_path=None, style_back_path=None, style_mask_path=None):
        self.condition_path = condition_path
        self.inference_image_dataset = inference_image_dataset
        self.initial_image_path = initial_image_path
        self.style_mask_path = style_mask_path

        self.styles = ["joker", "pixar", "sketch", "statue", "werewolf", "zombie"]

        #CHECK IF IT IS A INFERENCE BASED 
        if self.inference_image_dataset is not None:
            self.id_list = []
            self.inference_image_dataset_back_hair = os.path.join(self.inference_image_dataset, 'Back_Head')
            for img in sorted(os.listdir(self.inference_image_dataset_back_hair)):
                if img.endswith('.png') or img.endswith('.jpg'):
                    if train:
                        self.id_list.extend([img] * len(self.styles))  # Repeat for each style
                    else:
                        self.id_list.append(img)

        #else:      
        #    self.id_list = []
        #    for id_name in sorted(os.listdir(self.root+"_test")):
        #        self.id_list.append(id_name)
        #self.id_list = ['00012.jpg', '00023.jpg', '00048.jpg', '00052.jpg', '00067.jpg', 
        #'00103.jpg', '00115.jpg', '00116.jpg', '00122.jpg', '00125.jpg', '00133.jpg', 
        #'00138.jpg', '00140.jpg', '00143.jpg', '00145.jpg', '00153.jpg', '00167.jpg',
        # '00176.jpg', '00185.jpg', '00186.jpg', '00189.jpg', '00190.jpg', '00192.jpg',
        #  '00200.jpg', '00204.jpg', '00206.jpg', '00207.jpg', '00211.jpg', '00213.jpg',
        #   '00228.jpg', '00235.jpg', '00238.jpg', '00247.jpg', '00248.jpg', '00251.jpg',
        #    '00253.jpg', '00259.jpg', "00270.jpg"]
        self.transform = image_transform
        self.frame_rate = frame_rate = len([i for i in os.listdir(self.condition_path)])

        self.train = train

        
        self.stylized_img_path = style_front_path 
        self.stylized_img_back_path = style_back_path

    def __len__(self):
        return len(self.id_list) 

    def random_except(self, lst, excluded):
        filtered = [x for x in lst if x != excluded]
        if not filtered:
            raise ValueError("No elements left to choose from.")
        return random.choice(filtered)

    def __getitem__(self, idx):
        print("using_id:", self.id_list[idx])
        id_name = self.id_list[idx]

        style = self.styles[idx % len(self.styles)] if self.train else None
        print("style:", style)

        extra_appearance_list = []
        extra_appearance_list_style = []
        frame_rate = self.frame_rate
        #gt_path_ = os.path.join(self.inference_image_dataset, id_name, "image")
        if self.inference_image_dataset is not None:
            # Check the appearance flag and read the required number of images
            appearance = self.read_image(os.path.join(self.inference_image_dataset, 'input_image', id_name))
            if self.transform is not None:
                    condition_image = self.transform(appearance)
        condition_image_path_ = self.condition_path #os.path.join(self.root, 'conditions', 'sphere')#os.path.join(self.root, '0000000', "gt")
        gt_img_list = []
        drive_img_list = []

        if self.train:
            rand_id_name = self.random_except(self.id_list, id_name)

            if style is not None:
                condition_image_style = self.read_image(os.path.join(self.inference_image_dataset, style, "front_image_stylized", rand_id_name)) 
                if self.transform is not None:
                    condition_image_style = self.transform(condition_image_style)
            else:
                condition_image_style = None
        elif self.stylized_img_path is not None:
            condition_image_style = self.read_image(self.stylized_img_path)
            if self.transform is not None:
                condition_image_style = self.transform(condition_image_style)
        
        if self.initial_image_path is not None:
            fea_condition = []
            gt_view_path = f"{style}_{id_name.split('.')[0]}" if style is not None else id_name.split(".")[0]
            fea_condition_path = os.path.join(self.initial_image_path, gt_view_path, 'condition'+str(frame_rate)+'.mp4')
            print("using feature condition", fea_condition_path)
            cap = cv2.VideoCapture(fea_condition_path)
            frame_number = 0
            # Check if the video has been opened successfully
            if not cap.isOpened():
                print("Error opening video file")
            else:
                # Loop through each frame in the video
                while True:
                    ret, frame = cap.read()  # Read the frame
                    if not ret:
                        break  # Break the loop if there are no frames left to rea
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inital_image = self.transform(image)
                    fea_condition.append(inital_image)
                    # Increment the frame number
                    frame_number += 1

                # Release the video capture object
                cap.release()
            
        for i in range(frame_rate):               
            target_frame_idx = str(i).zfill(7)+'.png'
            #source_path = camera_folders[1]
            condition = self.read_image(os.path.join(condition_image_path_, target_frame_idx))
            
            if self.transform is not None:
                #target = self.transform(target)  
                condition = self.transform(condition)
            gt_img_list.append(condition)
            drive_img_list.append(condition)
            
            # if self.initial_image_path is not None:
            #     initial_image = condition

            #     id_name_ = id_name.split('.')[0]
            #     initial_image = self.read_image(os.path.join(self.initial_image_path, target_frame_idx))
            #     # if self.inference_image_dataset.split('_')[-2].endswith('sphere'):
            #     #     initial_image = self.read_image(os.path.join(self.initial_image_path, id_name_, target_frame_idx))
            #     # else:
            #     #     # if sequence 
            #     #     if i < 10 or i > 25:
            #     #         initial_image = self.read_image(os.path.join(self.inference_image_dataset, 'Front_Face', id_name))
            #     #     else:
            #     #         initial_image = self.read_image(os.path.join(self.inference_image_dataset, 'Generated_Face', id_name))
            #     print('reaqding noise from ',fea_condition_path )
            #     if self.transform is not None:
            #         initial_image = self.transform(initial_image)
            #     fea_condition.append(initial_image)
                
            # mask condition here
            
            #condition = np.ones_like(target).astype(np.uint8)
        prompt = '' #smile, open mouth, high quality'
        if fea_condition is not None:
            fea_condition = torch.stack(fea_condition)
        else:
            fea_condition = []
        res = {'condition_image': condition_image, 'condition': drive_img_list,'text_bg':prompt, 'text_blip': prompt, 'fea_condition': fea_condition, "image_name": id_name}
        #if self.more_image_control:
        #if id_name.endswith('.jpg'):
        #    id_name= id_name.split(".")[0]+".png"

        if self.stylized_img_path is not None or self.train:
            res['condition_image_style'] = condition_image_style

        more_appearance_frame = self.read_image(os.path.join(self.inference_image_dataset, 'Back_Head', id_name))
        #assert more_appearance_frame os.path.isfile(fname), f"File not found: {fname}"
        assert os.path.isfile(os.path.join(self.inference_image_dataset, 'Back_Head', id_name)), f"File not found: {os.path.join(self.inference_image_dataset, 'Back_Head', id_name)}"
        if self.transform is not None:
            more_appearance_frame = self.transform(more_appearance_frame) # we only consider
            extra_appearance_list.append(more_appearance_frame)
            #print("using edxtra_appera ce", os.path.join(self.inference_image_dataset, 'Back_Head', id_name))
        res['extra_appearance'] = more_appearance_frame  

        if self.train:
            more_appearance_frame_style = self.read_image(os.path.join(self.inference_image_dataset, style, 'back_image_stylized', rand_id_name))
            if self.transform is not None:
                more_appearance_frame_style = self.transform(more_appearance_frame_style)
                extra_appearance_list_style.append(more_appearance_frame_style)
                #print("using edxtra_appera ce", os.path.join(self.inference_image_dataset, style, 'Back_Head_stylized', id_name))     
            res['extra_appearance_style'] = more_appearance_frame_style
        elif self.stylized_img_back_path is not None:
            more_appearance_frame_style = self.read_image(self.stylized_img_back_path)
            if self.transform is not None:
                more_appearance_frame_style = self.transform(more_appearance_frame_style)
                extra_appearance_list_style.append(more_appearance_frame_style)
                #print("using edxtra_appera ce", os.path.join(self.inference_image_dataset, 'Back_Head_stylized', id_name))     

            res['extra_appearance_style'] = more_appearance_frame_style

        res['image'] = fea_condition

        if self.style_mask_path is not None:
            style_mask = self.read_image(self.style_mask_path, mask=True) if self.style_mask_path is not None else None
            style_mask = torch.from_numpy(style_mask).float()/255.0 if style_mask is not None else None
            res['style_mask'] = style_mask
            print("style_mask shape:", res['style_mask'].shape if res['style_mask'] is not None else "None")
        return res            
        
    def read_image(self, path, mask=False):
        #print("validation_image_path:", path)
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if not mask and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np


#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
from postproc.KNN import KNN

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs import point_cloud2
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
from sensor_msgs.msg import Image
import cv2

class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.split = split

        # ROS INIT
        rospy.init_node('pointcloud_inference', anonymous=True)
        self.header = Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = 'os_sensor'

        self.fields_xyz =[PointField('x',  0, 7, 1), # PointField.FLOAT32 = 7
                          PointField('y',  4, 7, 1),
                          PointField('z',  8, 7, 1),
                          PointField('intensity',  12, 7, 1)]
        
        self.fields =[PointField('x',  0, 7, 1), # PointField.FLOAT32 = 7
                 PointField('y',  4, 7, 1),
                 PointField('z',  8, 7, 1),
                 PointField('intensity',  12, 7, 1),
                 PointField('t', 16, 6, 1),
                 PointField('ring', 20, 2, 1)]
        
        self.point_type = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('t', np.uint32),
            ('ring', np.uint8)
        ])

        self.range_image_header = None
        self.signal_image_header = None

        self.sig_img = None
        self.range_img = None
        self.range_img_set = False
        self.sig_img_set = False
        
        # get the data
        from dataset.kitti.parser_ros import Parser
        self.parser = Parser(root=self.datadir,
                            #test_sequences=self.DATA["split"]["test"],
                            labels=self.DATA["labels"],
                            color_map=self.DATA["color_map"],
                            learning_map=self.DATA["learning_map"],
                            learning_map_inv=self.DATA["learning_map_inv"],
                            sensor=self.ARCH["dataset"]["sensor"],
                            max_points=self.ARCH["dataset"]["max_points"],
                            batch_size=1,
                            workers=self.ARCH["train"]["workers"],
                            gt=True,
                            shuffle_train=False)

        # concatenate the encoder and the head
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.model = HarDNet(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

    #     print(self.model)
        w_dict = torch.load(modeldir + "/SalsaNext_valid_best", map_location=lambda storage, loc: storage)
        self.model.load_state_dict(w_dict['state_dict'], strict=True)
        # use knn post processing?
        self.post = None
        
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes())
        print(self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def listener(self):
        # Subscribe to the input PointCloud2 topic
        #sc_lio_sam_global_map = '/sc_lio_sam/map_global'
        #sc_lio_sam_local_map = '/sc_lio_sam/map_local'
        
        self.model.eval()
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        
        ouster_points = '/ouster/points'
        ouster_rv = '/ouster/range_image'
        ouster_sig = '/ouster/signal_image'

        #rospy.Subscriber(ouster_points, PointCloud2, self.infer_pc)
        rospy.Subscriber(ouster_rv, Image, self.set_range_image)
        rospy.Subscriber(ouster_sig, Image, self.set_signal_image)

        # Create a publisher for the output PointCloud2 topic
        self.pub = rospy.Publisher('/semantic_points', PointCloud2, queue_size=10)

        rospy.spin()

    def publish_pc_xyz(self, points, labels):
        s_points = np.zeros((64*1024, 4), dtype=np.float32)  # [m, 3]: x, y, z
        s_points[:,0:3] = points.reshape(64*1024, 3)
        s_points[:,3] = labels
        labeled_cloud = pcl2.create_cloud(self.header, self.fields_xyz, s_points)
        labeled_cloud.is_dense = True  # Added line
        self.pub.publish(labeled_cloud)

    def set_signal_image(self, data):
        rospy.loginfo('Received a signal Image message')
        self.signal_image_header_seq = data.header.seq
        self.signal_image_header_stamp = data.header.stamp
        
        # Convert the byte data to a numpy array
        dtype = np.dtype(np.int16)  # as it's mono16
        dtype = dtype.newbyteorder('>')  # ROS Image messages use big endian
        cv_image = np.frombuffer(data.data, dtype=dtype).reshape(data.height, data.width)

        if cv_image.dtype.byteorder not in ('=', '|'):
            cv_image = cv_image.newbyteorder('=').astype(cv_image.dtype)

        self.sig_img = cv_image
        self.sig_img = self.sig_img.byteswap().newbyteorder() 

        self.infer_rv()
    
    def set_range_image(self, data):
        rospy.loginfo('Received a Range Image message')
        self.range_image_header_seq = data.header.seq
        self.range_image_header_stamp = data.header.stamp

        # Convert the byte data to a numpy array
        dtype = np.dtype(np.uint16)  # as it's mono16
        dtype = dtype.newbyteorder('>')  # ROS Image messages use big endian
        cv_image = np.frombuffer(data.data, dtype=dtype).reshape(data.height, data.width)

        if cv_image.dtype.byteorder not in ('=', '|'):
            cv_image = cv_image.newbyteorder('=').astype(cv_image.dtype)
        
        self.range_img = cv_image
        self.range_img = self.range_img.byteswap().newbyteorder()

        self.infer_rv()

    def infer_rv(self):
        if (self.range_image_header == self.signal_image_header):
            cnn = []
            knn = []        
            to_orig_fn=self.parser.to_original
            self.header.stamp = self.range_image_header_stamp # Make pointcloud stamp same as range image

            self.infer_subset(to_orig_fn, cnn=cnn, knn=knn)
        else:
            return
        
    def range_image_to_pointcloud(self, range_image, vertical_fov=(16.5, -16.5), horizontal_fov=(0, 360)):
        # Assuming range_image is a 2D numpy array
        height, width = range_image.shape

        # Create an array of angles (size same as range_image)
        theta = np.linspace(horizontal_fov[0], horizontal_fov[1], width)  # azimuth
        phi = np.linspace(vertical_fov[0], vertical_fov[1], height)  # elevation

        # Convert these to a grid of theta and phi values
        theta, phi = np.meshgrid(np.radians(theta), np.radians(phi))

        # Convert to Cartesian coordinates
        div = 250
        x = range_image * np.cos(phi) * np.sin(theta-np.pi/2) / div
        y = range_image * np.cos(phi) * np.cos(theta-np.pi/2) / div
        z = range_image * np.tan(phi) / div

        xyz = np.stack((x, y, z), axis=-1)#.reshape(-1,3)

        return xyz
   
    def infer_subset(self,to_orig_fn,cnn,knn):
        # empty the cache to infer in high res
        #if self.gpu:
        #    torch.cuda.empty_cache()
        
        with torch.no_grad():
            self.range_img = np.array(self.range_img, dtype=np.float32)
            self.sig_img = np.array(self.sig_img, dtype=np.float32)
            proj_range = torch.from_numpy(self.range_img).clone()

            unproj_range_np = self.range_img[:,:].flatten()
            unproj_range = torch.from_numpy(unproj_range_np).clone()

            indices = np.arange(unproj_range_np.shape[0])
            order = np.argsort(unproj_range_np)[::-1]
            indices = indices[order]

            proj_idx = np.full((64, 1024), -1, dtype=np.int32)
            proj_mask = np.zeros((64, 1024), dtype=np.int32)
            proj_idx[:, :] = indices.reshape(64, 1024)
            proj_mask = (proj_idx > 0).astype(np.int32)
            proj_mask = torch.from_numpy(proj_mask)

            height, width = self.range_img.shape
            p_y, p_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            p_x = torch.from_numpy(p_x.flatten()).clone()
            p_y = torch.from_numpy(p_y.flatten()).clone()

            unproj_xyz = self.range_image_to_pointcloud(self.range_img)
            proj_xyz_np = np.full((64, 1024, 3), -1, dtype=np.float32)
            proj_xyz_np[:,:] = unproj_xyz[:]
            
            proj_remission = torch.from_numpy(self.sig_img).clone()
            proj_xyz = torch.from_numpy(proj_xyz_np).clone()

            proj_in = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()]).float()

            img_means = torch.tensor([[[ np.mean(unproj_range_np)]],
                                  [[ np.mean(proj_xyz_np[:,:,0])]],
                                  [[np.mean(proj_xyz_np[:,:,1])]],
                                  [[ np.mean(proj_xyz_np[:,:,2])]],
                                  [[ np.mean(self.sig_img[:,:])]]])
            
            img_stds = torch.tensor([[[ np.std(unproj_range_np)]],
                                  [[ np.std(proj_xyz_np[:,:,0])]],
                                  [[np.std(proj_xyz_np[:,:,1])]],
                                  [[ np.std(proj_xyz_np[:,:,2])]],
                                  [[ np.std(self.sig_img[:,:])]]])
            
            proj_in = (proj_in - img_means) / img_stds
            proj_in = proj_in * proj_mask.float()

            proj_in = proj_in.float()
            
            #cv2.imshow("Model image", proj_range.numpy())
            #key = cv2.waitKey(100) & 0xFF

            if self.gpu:
                proj_in = proj_in.cuda()
                p_x = p_x.cuda()
                p_y = p_y.cuda()
            if self.post:
                proj_range = proj_range.cuda()
                unproj_range = unproj_range.cuda()
            end = time.time()

            if self.ARCH["train"]["aux_loss"]:
                with torch.cuda.amp.autocast(enabled=True):
                    [proj_output, x_2, x_3, x_4] = self.model(proj_in.unsqueeze(0))
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    proj_output = self.model(proj_in)
                    
            proj_argmax = proj_output[0].argmax(dim=0)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
    #        print("Network seq", path_seq, "scan", path_name, "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range, unproj_range, proj_argmax, p_x, p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            
            #print(f"KNN Infered point cloud range view in {res} sec")
            knn.append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            pred_np = to_orig_fn(pred_np)
            #print(f"predictions numpy:\n {pred_np}")
            
            self.publish_pc_xyz(unproj_xyz, pred_np)
            #self.publish_pc(pred_np)
#!/usr/bin/python

##### (Base code)
#
## Massimiliano Patacchiola, Plymouth University 2016
#
# This code uses Self-Organizing Map (SOM) to classify different poses (pan, tilt) of a humanoid robot (NAO).
#
# ------------------------------------------------------------
###### (Modifications)
#
## Marco Flores-Corondo, Universidad Autónoma del Estado de Morelos 2020
#
#
# The SOM instance inside the pyERA has been modified; saveplain attribute has been added to save 
# self._weights_matrix as a .txt file. Pleade add to the SOM instance the following and re-install pyERA
#
#    def saveplain(self,path="./", name="som"):
 #       outfile=path+name+".txt"
 #       data=[]
 #       for i in range(self._matrix_size):
 #           for j in range(self._matrix_size):
 #               temporal=self.get_unit_weights(i,j)
 #               data.append(temporal)
 #       np.savetxt(outfile,data,delimiter=",")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
import time

#It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay
from pyERA.utils import LinearDecay

USE_NAO = False #if True connect to the NAO
NAO_IP = "192.168.0.100"
NAO_PORT = 9559
effectorName="RArm"
# uncomment to declare pynaoqi library direction or write the following comand before running the script:
# export PYTHONPATH={PYTHONPATH}:location/of/pynaoqi
if(USE_NAO == True):
    import sys
    #sys.path.insert(1, "location/of/pynaoqi")  #optional line
    from naoqi import ALProxy




def main():  

    if(USE_NAO == True):
        print("Foetus engineer will Init the Nao Robot...")
        _al_motion_proxy = ALProxy("ALMotion", NAO_IP, int(NAO_PORT))
        _al_posture_proxy = ALProxy("ALRobotPosture", NAO_IP, int(NAO_PORT))
        _al_posture_proxy.goToPosture("Crouch", 0.5)
        effectorInit = _al_motion_proxy.getPosition(effectorName, space, useSensor)
        space= _al_motion_proxy.FRAME_ROBOT
        useSensor = False
       
        time.sleep(3.0)
        _al_motion_proxy.wbEnableEffectorControl(effectorName, USE_NAO)
        time.sleep(2.0)
        print("Johny Nazco is starting the training...")
    
    #Set to True if you want to save the SOM images inside a folder.
    SAVE_IMAGE = True
    #Set to True if you want to save every SOM every 10% of training as plain text.
    SAVE_PLAIN= False
    output_path = "./output/" #Change this path to save in a different folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Init the SOM
    som_size = 16
    my_som = Som(matrix_size=som_size, input_size=3, low=-1, high=+1, round_values=False)  

    #Init the parameters
    tot_epoch = 100
    my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=tot_epoch/5, decay_rate=0.9, staircase=True)
    my_radius = ExponentialDecay(starter_value=np.rint(som_size/3), decay_step=tot_epoch/6, decay_rate=0.90, staircase=True)

    #Starting the Learning
    for epoch in range(1, tot_epoch+1):

        #Saving the image associated with the SOM weights
        if(SAVE_IMAGE == True):
            if ((epoch%(tot_epoch/10)) ==0) or (epoch==tot_epoch+1):
                save_path = output_path + str(epoch) + ".png"
                img = my_som.return_weights_matrix()
                plt.axis("off")
                plt.imshow(img,cmap="rainbow", vmin=-1,vmax=1)
                plt.savefig(save_path)
               
        if(SAVE_PLAIN==True):
            if ((epoch%(tot_epoch/10)) ==0)
                my_som.saveplain(path=output_path, name= effector+"_som_babbling"+str(epoch))

        #Updating the learning rate and the radius
        learning_rate = my_learning_rate.return_decayed_value(global_step=epoch)
        radius = my_radius.return_decayed_value(global_step=epoch)

        #Generating random input vectors
        if (effectorName=="LArm"):
            xAxis=round(np.random.uniform(0,0.12),4)
            yAxis=round(np.random.uniform(-0.5,0.10),4)
            zAxis=round(np.random.uniform(-0.10,0.10),4)
        elif (effectorName=="RArm"):
            xAxis=round(np.random.uniform(0,0.12),4)
            yAxis=round(np.random.uniform(-0.10,0.5),4)
            zAxis=round(np.random.uniform(-0.10,0.10),4)
        target= np.array([xAxis, yAxis, zAxis])
        if (USE_NAO==True):
            input_vector=np.aray([target[i] + effectorInit[i] for i in range(3)],dtype=np.float32)
        else:
            input_vector=np.array([target[i]for i in range(3)], dtype=np.float32)
        # Move Robot to target position
        if(USE_NAO == True):
            _al_motion_proxy.wbEnableEffectorControl(effectorName, USE_NAO)
            print("Elmer Ofeto is feeding these angles to NAO: " + str(input_vector))
            _al_motion_proxy.wbSetEffectorControl(effectorName, input_vector)
            time.sleep(4.0) #Get time to NAO to reach the point

        #Estimating the BMU coordinates
        bmu_index = my_som.return_BMU_index(input_vector)
        bmu_weights = my_som.get_unit_weights(bmu_index[0], bmu_index[1])

        #Getting the BMU neighborhood
        bmu_neighborhood_list = my_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)  

        #Learning step      
        my_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=True)

        print("")
        print("Epoch: " + str(epoch))
        print("Learning Rate: " + str(learning_rate))
        print("Radius: " + str(radius))
        print("Input vector: " + str(input_vector))
        print("BMU index: " + str(bmu_index))
        print("BMU weights: " + str(bmu_weights))
        #print("BMU NEIGHBORHOOD: " + str(bmu_neighborhood_list))

    #Reset the NAO head
    if(USE_NAO == True):
        print("Resetting NAO")
        isEnabled    = False
        _al_motion_proxy.wbEnableEffectorControl(effectorName, isEnabled)
        time.sleep(4.0)

    #Saving the network
    file_name = output_path + "som_babbling.npz"
    text_file= output_path + effector+"_som_babbling.txt"
    print("Saving the network in: " + str(file_name))
    my_som.save(path=output_path, name="som_babbling")
    print("Saving as plain text in:"+str(text_fle))
    my_som.saveplain(path=output_path, name= effector+"_som_babbling"+str(epoch))


    #img = np.rint(my_som.return_weights_matrix())
    #plt.axis("off")
    #plt.imshow(img)
    #plt.show()

if __name__ == "__main__":
    
    main()

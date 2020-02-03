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

def  save_map_image(save_path, size, weight_matrix, yaw_max_range=2.0, pitch_max_range=2.0):
    x = np.arange(0, size, 1) + 0.5
    y = np.arange(0, size, 1) + 0.5

    fig = plt.figure()
    #plt.title('NAO Head Pose SOM')
    ax = fig.gca()
    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.set_xticks(np.arange(1, size+1, 1))
    ax.set_yticks(np.arange(1, size+1, 1))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

    for line in ticklines:
        line.set_linewidth(2)

    for line in gridlines:
        line.set_linestyle('-')
        line.set_color("grey")


    tot_rows = weight_matrix.shape[0]
    tot_cols = weight_matrix.shape[1]

    for row in range(0, tot_rows):
        for col in range(0, tot_cols):
            yaw=weight_matrix[row,col,0]
            pitch=weight_matrix[row,col,1]
            if(pitch > 30.0 or pitch < -30): pitch_max_range=90.0
            else: pitch_max_range = 30.0
            yaw_arrow = (yaw / yaw_max_range) * 0.4
            pitch_arrow = (pitch / pitch_max_range) * 0.4
            ax.arrow(row+0.5, col+0.5, yaw_arrow, pitch_arrow, head_width=0.1, head_length=0.1, fc='k', ec='k')


    #s is the dot area and c is the color
    #plt.scatter(x, y, s=3.0, c="black")
    #plt.grid()
    #plt.show()
    ax.axis('off')
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close('all')


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
    output_path = "./output/" #Change this path to save in a different forlder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #Init the SOM
    som_size = 16
    my_som = Som(matrix_size=som_size, input_size=3, low=-2, high=+2, round_values=False)  

    #Init the parameters
    tot_epoch = 100
    my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=tot_epoch/5, decay_rate=0.9, staircase=True)
    my_radius = ExponentialDecay(starter_value=np.rint(som_size/3), decay_step=tot_epoch/6, decay_rate=0.90, staircase=True)

    #Starting the Learning
    for epoch in range(1, tot_epoch+1):

        #Saving the image associated with the SOM weights
        if(SAVE_IMAGE == True):
            if (((epoch*100)/tot_epoch)%10 ==0) or (epoch==tot_epoch+1):
                save_path = output_path + str(epoch) + ".png"
                save_map_image(save_path, som_size, my_som.return_weights_matrix())
            

        #Updating the learning rate and the radius
        learning_rate = my_learning_rate.return_decayed_value(global_step=epoch)
        radius = my_radius.return_decayed_value(global_step=epoch)

        #Generating random input vectors
        xAxis=round(np.random.uniform(0,0.12),2)
        yAxis=round(np.random.uniform(-0.5,0.10),2)
        zAxis=round(np.random.uniform(-0.10,0.10),2)
        if (effectorName == "LArm"):
                coef = +1.0
        elif (effectorName == "RArm"):
                coef = -1.0
        target= np.array([xAxis, yAxis*coef, zAxis])
        target=[axis*np.pi/180 for axis in target]
        if (USE_NAO==True):
            input_vector=np.aray([target[i] + effectorInit[i] for i in range(3)],dtype=np.float32)
        else:
            input_vector=np.array([target[i]for i in range(3)], dtype=np.float32)
        # Move Robot to target position
        if(USE_NAO == True):
            _al_motion_proxy.wbEnableEffectorControl(effectorName, USE_NAO)
            print("Elmer Ofeto is feeding these angles to NAO: " + str(input_vector))
            _al_motion_proxy.wbSetEffectorControl(effectorName, input_vector)
            time.sleep(3.0) #Get time to NAO to reach the point

        #Estimating the BMU coordinates
        bmu_index = my_som.return_BMU_index(input_vector)
        bmu_weights = my_som.get_unit_weights(bmu_index[0], bmu_index[1])

        #Getting the BMU neighborhood
        bmu_neighborhood_list = my_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius=radius)  

        #Learning step      
        my_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)

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
        print("[PYERA] Reset NAO head...")
        _al_motion_proxy.setAngles("HeadPitch", 0, 0.3)
        _al_motion_proxy.setAngles("HeadYaw", 0, 0.3)
        time.sleep(2.0)

    #Saving the network
    file_name = output_path + "som_babbling.npz"
    text_file= output_path + "som_RArm_babbling.txt"
    print("Saving the network in: " + str(file_name))
    my_som.save(path=output_path, name="som_babbling")
    print("Saving as plain text in:"+str(text_fle))
    my_som.saveplain(path=output_path, name= "som_lArm_babbling")


    #img = np.rint(my_som.return_weights_matrix())
    #plt.axis("off")
    #plt.imshow(img)
    #plt.show()

if __name__ == "__main__":
    
    main()

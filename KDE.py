#Alexander Pavlak
#CSE 4334-001
#Homework 1: Kernel Density Estimation

import numpy as np
import matplotlib.pyplot as plt 

def mykde(X,h):
    #determine domain
    domain = np.array([np.amin(X),np.amax(X)])
    #In order to compute we must discretize the domain
    partitionSize = .01
    binLocation = domain[0] + partitionSize
    numberOfBins = int(domain[1] / partitionSize)
    #create the array for probabilities
    probabilities= np.zeros((numberOfBins,2))
    binCounter = 0
    while(binCounter < numberOfBins):
        #count how many points occur in the bin
        pointsInBin = 0
        for i in X:
            
            distanceToBin = ((binLocation - i) /h)
            if(abs(distanceToBin) <= .5):
                pointsInBin +=1
            
        probabilities[binCounter] = ([(1/(len(X)*(h**X.ndim))) * pointsInBin,binLocation])
        #print(probabilities) 
        binCounter += 1 
        binLocation += partitionSize
    returnProbs = np.array(probabilities)
    returnProbs = np.delete(returnProbs,(0),axis=0)
    return returnProbs, domain

def mykde2D(X,h):
    #determine domain
    xDomain = np.array(([np.amin(X[:,0]),np.amax(X[:,0])]))
    yDomain = np.array(([np.amin(X[:,1]),np.amax(X[:,1])]))
    #In order to compute we must discretize the domain
    xPartitionSize = abs(xDomain[0]) + abs(xDomain[1])
    yPartitionSize = abs(yDomain[0]) + abs(yDomain[1])
    #partition based on X value
    numberOfXBins = 5
    numberOfYBins = 5
    xPartitionSize /= numberOfXBins
    yPartitionSize /= numberOfYBins
    #create the array for probabilities
    probabilities= np.zeros(((numberOfXBins+1)*numberOfYBins,3))
    loopCounter = 0
    xPoint = xDomain[0]
    while(loopCounter < (len(probabilities))):
        yPoint = yDomain[0]
        #count how many points occur in the bin
        for j in range(numberOfYBins):

            pointsInBin = 0
            for k in X:
                point = np.array((xPoint,yPoint))
                distanceToBin = ((np.linalg.norm(point - k)) /h)
                if(abs(distanceToBin) <= .5):
                    pointsInBin +=1
            yPoint += yPartitionSize
            probabilities[loopCounter,0] = xPoint
            probabilities[loopCounter,1] = yPoint
            probabilities[loopCounter,2] = ((1/(len(X)*(h**X.ndim))) * pointsInBin)
            loopCounter += 1 

        
        xPoint += xPartitionSize
        
    #get output ready to be returned  
    returnProbs = np.array(probabilities)
    returnProbs = np.delete(returnProbs,(0),axis=0)
    domain = np.concatenate((xDomain,yDomain),axis=0)
    return returnProbs, domain


###MAIN###

# Experiment 1: Non clustered data
set1 = np.random.normal(5,1,1000)
h = np.array([.1,1,5,10])

#Hardcoded running of the KDE and plotting to work with subplots.
fig, axs = plt.subplots(2,2)

subTitle = 'h =' + str(h[0])
p, x = mykde(set1,h[0])
axs[0,0].scatter(p[:,1],p[:,0],s=2,c='g')
axs[0,0].hist(set1, 20,alpha=.5,density=True)
axs[0,0].set_title(subTitle)

subTitle = 'h =' + str(h[1])
p, x = mykde(set1,h[1])
axs[0,1].scatter(p[:,1],p[:,0],s=2,c='g')
axs[0,1].hist(set1, 20,alpha=.5,density=True)
axs[0,1].set_title(subTitle)

subTitle = 'h =' + str(h[2])
p, x = mykde(set1,h[2])
axs[1,0].scatter(p[:,1],p[:,0],s=2,c='g')
axs[1,0].hist(set1, 20,alpha=.5,density=True)
axs[1,0].set_title(subTitle)

subTitle = 'h =' + str(h[3])
p, x = mykde(set1,h[3])
axs[1,1].scatter(p[:,1],p[:,0],s=2,c='g')
axs[1,1].hist(set1, 20,alpha=.5,density=True)
axs[1,1].set_title(subTitle)

plt.suptitle('Non Clustered Data')
plt.tight_layout()
plt.show()

#Experiment 2: Clustered data
set2 = np.random.normal(0,0.2,1000)
combined = np.concatenate((set1,set2))

#Hardcoded running of the KDE and plotting to work with subplots.
fig, axs = plt.subplots(2,2)

subTitle = 'h =' + str(h[0])
p, x = mykde(combined,h[0])
axs[0,0].scatter(p[:,1],p[:,0],s=2,c='g')
axs[0,0].hist(combined, 20,alpha=.5,density=True)
axs[0,0].set_title(subTitle)

subTitle = 'h =' + str(h[1])
p, x = mykde(combined,h[1])
axs[0,1].scatter(p[:,1],p[:,0],s=2,c='g')
axs[0,1].hist(combined, 20,alpha=.5,density=True)
axs[0,1].set_title(subTitle)

subTitle = 'h =' + str(h[2])
p, x = mykde(combined,h[2])
axs[1,0].scatter(p[:,1],p[:,0],s=2,c='g')
axs[1,0].hist(combined, 20,alpha=.5,density=True)
axs[1,0].set_title(subTitle)

subTitle = 'h =' + str(h[3])
p, x = mykde(combined,h[3])
axs[1,1].scatter(p[:,1],p[:,0],s=2,c='g')
axs[1,1].hist(combined, 20,alpha=.5,density=True)
axs[1,1].set_title(subTitle)

plt.suptitle('Clustered Data')
plt.tight_layout()
plt.show()

#Experiment 3: 2D data

#Given parameters for generating 1st set of gausian numbers
mean1 = np.array([1,0])
dev1 = np.array([[0.9,0.4],[0.4,0.9]])
#Given parameters for generating 2nd set of gausian numbers
mean2 = np.array([0,1.5])
dev2 = ([[0.9,0.4],[0.4,0.9]])
#create clusters and combine them into a single set
set2D1 = np.random.multivariate_normal(mean1,dev1,500)
set2D2 = np.random.multivariate_normal(mean2,dev2,500)
X = np.concatenate((set2D1,set2D2),axis=0)

fig2, axs2 = plt.subplots(2,2)

subTitle = 'h =' + str(h[0])
p,x = mykde2D(X,h[0])
volume = p[:,2]
#scale the probabilities to make meaningful values
volume *= 10000
axs2[0,0].scatter(X[:,0],X[:,1])
axs2[0,0].scatter(p[:,0],p[:,1],c='g',alpha=.9,s=volume)
axs2[0,0].set_title(subTitle)

subTitle = 'h =' + str(h[1])
p,x = mykde2D(X,h[1])
volume = p[:,2]
#scale the probabilities to make meaningful values
volume *= 10000
axs2[0,1].scatter(X[:,0],X[:,1])
axs2[0,1].scatter(p[:,0],p[:,1],c='g',alpha=.9,s=volume)
axs2[0,1].set_title(subTitle)

subTitle = 'h =' + str(h[2])
p,x = mykde2D(X,h[2])
volume = p[:,2]
#scale the probabilities to make meaningful values
volume *= 10000
axs2[1,0].scatter(X[:,0],X[:,1])
axs2[1,0].scatter(p[:,0],p[:,1],c='g',alpha=.9,s=volume)
axs2[1,0].set_title(subTitle)

subTitle = 'h =' + str(h[3])
p,x = mykde2D(X,h[3])
volume = p[:,2]
#scale the probabilities to make meaningful values
volume *= 10000
axs2[1,1].scatter(X[:,0],X[:,1])
axs2[1,1].scatter(p[:,0],p[:,1],c='g',alpha=.9,s=volume)
axs2[1,1].set_title(subTitle)

plt.suptitle('2D Data')
plt.tight_layout()
plt.show()

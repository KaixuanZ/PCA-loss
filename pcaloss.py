import caffe
import numpy as np
from scipy.linalg.decomp import eigh



class PCALossLayer(caffe.Layer):

    def setup(self, bottom, top):
        """ Constructor """
        # check input data
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute PCA space.")
        self.classes = []
        self.evals=[]
        self.evecs=[]
        self.n_components = None
        self.threshold=0.01
        self.backpro=0

    def reshape(self, bottom, top):
        # difference is shape of inputs
        #bottom[0].diff = np.zeros(bottom[0].data.shape, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.setup(bottom,top)
        # each person should be a row in bottom[0].data
        if(len(bottom[0].data.shape)==4):
            data=bottom[0].data[:,:,0,0]
        else:
            data=bottom[0].data
        if(len(bottom[1].data.shape)==4):
            y=bottom[1].data[:,0,0,0]
        else:
            y=bottom[1].data
        assert data.shape[0]==y.shape[0]

        # get class labels
        classes = np.unique(y)

        # compute covs
        covs = []
        for group in classes:
            Xg = data[y == group, :]
            if Xg.shape[0]>1:
                self.backpro=1
                covs.append(np.cov(Xg.T))
                self.classes.append(group)

        # within scatter
        if self.backpro:
            Sw = np.average(covs, axis=0)

            # compute eigen decomposition and evals are in an ascending order
            evals, evecs= np.linalg.eigh(Sw)
            thresh = evals.max()*self.threshold

            index=(evals >= thresh).nonzero()
            evals = evals[index[0]]
            evecs = evecs[:,index[0]]            
            self.evals=evals
            self.evecs=evecs
            self.n_components=len(self.evals)
            #minimize the loss
            top[0].data[...] = np.average(self.evals, axis=0)

        else:
            top[0].data[...]=0

    def backward(self, top, propagate_down, bottom):
        if propagate_down:
            if(len(bottom[0].data.shape)==4):
                data=bottom[0].data[:,:,0,0]
            else:
                data=bottom[0].data
            if(len(bottom[1].data.shape)==4):
                y=bottom[1].data[:,0,0,0]
            else:
                y=bottom[1].data
            batch_size,width=data.shape
            Sw_Hij=np.zeros([width,width])

            for i in range(batch_size):
                if y[i] in self.classes:
                    for j in range(width):
                        diff_sum=0
                        Sw_Hij[...]=0

	            	#compute Sw_Hij
                        data_c=data[y==y[i],:]
                        Sw_Hij[j,:]=data[i,:]-data_c.mean(0)
                        Sw_Hij[:,j]+=Sw_Hij[j,:]
                        Sw_Hij=Sw_Hij*1.0/len(self.classes)/(data_c.shape[0]-1)
                        assert Sw_Hij.all()==Sw_Hij.T.all(),"St_Hij should be symmetric"

                        for k in range(self.n_components):
                            diff=Sw_Hij
                            diff=np.dot(self.evecs[:,k].T,diff)
                            diff=np.dot(diff,self.evecs[:,k])	#gradient of a single eigenvalue
                            diff_sum+=diff

                        #the grad of Hij
                        bottom[0].diff[i,j]=top[0].diff*diff_sum/self.n_components
        else:
            bottom[0].diff[...]=0;

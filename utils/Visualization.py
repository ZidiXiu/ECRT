import numpy as np
import os
import matplotlib.pyplot as plt

def show_img(X,M=5,x0=[],size=[28,28]):
    
    '''
    show_img(X,M=5,x0=[],size=[28,28])
    
    '''
    
    h = size[0]
    w = size[1]
    
    if x0==[]:
        x0 = np.reshape(X[0,:],[h,w])

    nb_nei = np.shape(X)[0]

    if X.dtype == 'uint8':
        all_imgs = np.zeros([h*M,w*M],dtype=np.uint8)
    else:
        all_imgs = np.zeros([h*M,w*M])
    
    for k in range(M**2):

        jj = k%M
        ii = np.int((k-jj)/M)

        if k>=nb_nei:
            break

        x = X[k]
        if k==0:
            x = x0

        all_imgs[h*ii:h*(ii+1),w*jj:w*(jj+1)] = np.reshape(x,[h,w])

    _ = plt.imshow(all_imgs,cmap='gray')
    _ = plt.title('Neighbors %d' % (nb_nei))
    
def draw_grad(Txy,s,scal,head_width,head_length,fc='g',ec='g'):
    ax = plt.gca()
    _ = plt.axis('square')
    for j in range(np.shape(Txy)[0]):
        ax.arrow(Txy[j,0], Txy[j,1], scal*s[j,0], scal*s[j,1], \
                 head_width=head_width, head_length=head_width, fc=fc, ec=ec)
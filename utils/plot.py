import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def visualize_space(s_space, label_, source_name, torch_object=True, model_name='', plot_path = None):
    sns.set_palette('Set2')
    if torch_object:
        s_space = s_space.detach().cpu().numpy()
        
    
    nlabel = len(np.unique(label_))
#     sns.set_palette("Spectral", n_colors=nlabel)
    
    df = pandas.DataFrame({'s_x':s_space[:,0], 's_y':s_space[:,1], 'label':label_})
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='s_x', y='s_y', hue='label', data=df, alpha=0.6, legend=False, palette="Set2")
    plt.xlabel(source_name+'_1')
    plt.ylabel(source_name+'_2')
    
    plt.title(source_name+' space')
    plt.show()
    
    if type(plot_path) != type(None):
        plt.savefig(plot_path+'/' +  model_name + '_'+source_name+'.pdf', bbox_inches='tight')
        plt.clf()

def visualize_s_z_space(s_space, z_space, label_s, label_z=None, name_list=None,torch_object = True, model_name='', plot_path = None):
    sns.set_palette('Set2')
    if type(label_z) == type(None):
        label_z = label_s.copy()
    if torch_object:
        s_space = s_space.detach().numpy()
        z_space = z_space.detach().numpy()
        label_s = label_s.detach().numpy()
        label_z = label_z.detach().numpy()
        
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    
    df = pandas.DataFrame({'s_x':s_space[:,0], 's_y':s_space[:,1], 'label':label_s})
    sns.scatterplot(x='s_x', y='s_y', hue='label', data=df, alpha=0.6, palette="Set2", legend=False)
    plt.xlabel(name_list[0]+'_1')
    plt.ylabel(name_list[0]+'_2')
#     plt.xlim(-6,10)
#     plt.ylim(-6,10)

    plt.title(name_list[0]+' space')
    plt.subplot(1,2,2)
    df = pandas.DataFrame({'s_x':z_space[:,0], 's_y':z_space[:,1], 'label':label_z})

    sns.scatterplot(x='s_x', y='s_y', hue='label', data=df, alpha=0.6, palette="Set2", legend=False)
    plt.xlabel(name_list[1]+'_1')
    plt.ylabel(name_list[1]+'_2')

#     plt.xlim(-6,10)
#     plt.ylim(-6,10)    

    plt.title(name_list[1]+' space')
    plt.show()
    
    if type(plot_path) != type(None):
        plt.savefig(plot_path+'/' +  model_name + '.pdf', bbox_inches='tight')
        plt.clf()
       
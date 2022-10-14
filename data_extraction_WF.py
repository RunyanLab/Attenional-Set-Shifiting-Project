from wfield import *
import os 
import h5py
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter

sync = "sync"
path = os.path.join(directory) #general path 

path_sync = os.path.join(directory, sync) #path for sync files + df 
os.chdir(path_sync)
df = pd.read_csv('All_data_imaged.csv')

path_sesh = os.path.join(path, str(img_session)) #path for single imaging session 
os.chdir(path_sesh)

def landmarks():
    lmarks = load_allen_landmarks('dorsal_cortex_landmarks.json')

    ccf_regions_reference,proj,brain_outline = allen_load_reference('dorsal_cortex')
    # the reference is in allen CCF space and needs to be converted
    # this converts to warped image space (accounting for the transformation)
    ccf_regions = allen_transform_regions(None,ccf_regions_reference,
                                          resolution = lmarks['resolution'],
                                            bregma_offset = lmarks['bregma_offset'])
    atlas, areanames, brain_mask = atlas_from_landmarks_file('dorsal_cortex_landmarks.json') # this loads the atlas in transformed coords

    # this does the transform (warps the original images)
    stack.set_warped(1, M = lmarks['transform']) # this warps the spatial components in the stack

    # this converts the reference to image space (unwarped)
    atlas_im, areanames, brain_mask = atlas_from_landmarks_file('dorsal_cortex_landmarks.json',do_transform = True) # this loads the untransformed atlas
    ccf_regions_im = allen_transform_regions(lmarks['transform'],ccf_regions_reference,
                                            resolution = lmarks['resolution'],
                                            bregma_offset = lmarks['bregma_offset'])
    
    return atlas, atlas_im, ccf_regions_im

# extracts raw ROI activity from SVD 
def extract_raw_ROI(path, img_session):
    path_sesh = os.path.join(path, str(img_session))
    os.chdir(path_sesh)
    print(path_sesh)
    U = np.load('U.npy')
    SVT = np.load('SVTcorr.npy')
    stack = SVDStack(U,SVT)
    
    landmarks()
    stack.set_warped(True) # once this is done once the transform is set and you can alternate between the 2 modes.
    activityR = []
    activityL = []
    for area in range(33):
        right = stack.get_timecourse(np.where(atlas == area)).mean(axis = 0)
        left = stack.get_timecourse(np.where(atlas == -area)).mean(axis = 0)
        activityR.append(right*100)
        activityL.append(left*100)
    activityR = np.array(activityR, dtype=object)
    activityL = np.array(activityL, dtype=object)
    #save activity matrices for easier access 

    np.save('activityR', activityR)
    np.save('activityL', activityL)
    
    return activityR, activityL

# takes raw ROI activity and aligns it to entire session  
def get_sesh_actvity(path, img_session, right_left): #right_left 0 = right and right_left 1 = left 
    path_sesh = os.path.join(path, str(img_session))
    path_sync = os.path.join(path, "sync")
    os.chdir(path_sesh)
    U = np.load('U.npy')
    SVT = np.load('SVT.npy')
    stack = SVDStack(U,SVT)
    
    landmarks()
    
    if right_left == 0:
        activity_zscored = np.load('activityR.npy', allow_pickle=True)
    if right_left == 1:
        activity_zscored = np.load('activityL.npy', allow_pickle=True)
        
    #load z-scored activity matrices
    activity_zscored = savgol_filter(activity_zscored, 6, 1)
    os.chdir(path_sync)
    df = pd.read_csv('All_data_imaged.csv')
    df_sesh = df[df['img session'] == img_session]
    df2 = df_sesh[df_sesh['start frame'] > 0].reset_index(drop=True)
    df2 = df2.drop(['Unnamed: 0'], axis=1)
    x = df2.index.tolist()
    last_val = x[-1]
    
    sesh_activity = []
    for i in x:
        trial = i
        if trial < last_val:
            start = df2['start frame'][trial] 
            stop =  df2['start frame'][trial+1]
        elif trial == last_val:
            start = df2['start frame'][trial]
            stop = 9001
        trial_activity = []
        if start < 0:
            area = list(range(33))
            for x in area:
                trial_activity.append(0)
        elif start > 0:
            area = list(range(33))
            for x in area:
                trial_activity.append(activity_zscored[x][start:stop])
        sesh_activity.append(trial_activity)
    sesh_activity = np.array(sesh_activity, dtype=object) #indexing: sesh_activityR[trial][ROI]

    return sesh_activity

# trializes the session activity 
def trialized_activity(activity):
    path_sesh = os.path.join(path, str(img_session))
    path_sync = os.path.join(path, "sync")
    os.chdir(path_sync)
    df2 = pd.read_csv('All_data_imaged.csv')
    os.chdir(path_sesh)
    full_activity_all = []
    trial_activity_all =[]
    ITI_activity_all = []

    for trial_index in df2.index.tolist():

        trial_start = df2['start frame'][trial_index]
        trial_end = df2['end frame'][trial_index]
        ITI_start = trial_end-trial_start
        if df2['correct'][trial_index] == 1:
            ITI_end = ITI_start + 75
        elif df2['correct'][trial_index] == 0:
            ITI_end = ITI_start + 150

        full_activity_ROI = []
        trial_activity_ROI = []
        ITI_activity_ROI = []

        for area in list(range(33)):
            full_activity = activity[trial_index][area][0:ITI_end]
            trial_activity = activity[trial_index][area][0:ITI_start] #to get trial activity [0:ITI_start]
            ITI_activity = activity[trial_index][area][ITI_start:ITI_end] #to get ITI activity [ITI_start:ITI_end] 

            full_activity_ROI.append(full_activity)
            trial_activity_ROI.append(trial_activity)
            ITI_activity_ROI.append(ITI_activity)

        trial_activity_all.append(trial_activity_ROI*10)
        ITI_activity_all.append(ITI_activity_ROI*10)
        full_activity_all.append(full_activity_ROI*10)

    full_activity_all = np.array(full_activity_all, dtype=object) #indexing trial_activity_all[trial][area]
    trial_activity_all = np.array(trial_activity_all, dtype=object) #indexing trial_activity_all[trial][area]
    ITI_activity_all = np.array(ITI_activity_all, dtype=object) #indexing ITI_activity_all[trial][area]
    
    return full_activity_all, trial_activity_all, ITI_activity_all

def round_act(act):
    numElems = 200
    act_rnd_all = []
    for ROI in range(33):
        ind = act.shape[1]
        act_rnd = []
        for trial in range(ind):
            x = act[1,trial,ROI]
            len_act = len(x)
            rnd = np.round(np.linspace(0, len_act-1, numElems)).astype(int)
            act_rnd.append(x[rnd])
        act_rnd_all.append(act_rnd)
    act_rnd_all = np.array(act_rnd_all, dtype=object)
    act_rnd_all = act_rnd_all.transpose(1,0,2)
    return act_rnd_all

#Run for specific imaging session 

for i in list(range(1,num_files+1)):
    extract_raw_ROI(path, i)
print('done')

os.chdir(path)
group_actR = np.concatenate([get_sesh_actvity(path, img_session = i, right_left = 0) for i in range(1,num_files+1)])
print('done right')
os.chdir(path_sync)
np.save('group_actR.npy', group_actR)

os.chdir(path)
group_actL = np.concatenate([get_sesh_actvity(path, img_session = i, right_left = 1) for i in range(1,num_files+1)])
print('done left')
os.chdir(path_sync)
np.save('group_actL.npy', group_actL)

#trialize activity 
T_R = trialized_activity(group_actR)
T_R = np.array(T_R, dtype=object)

T_L = trialized_activity(group_actL)
T_L = np.array(T_L, dtype=object)

#subsample 200 points from all trials 
act_R = round_act(T_R)
act_L = round_act(T_L)

os.chdir(path_sync)
np.save('act_R.npy', act_R)
np.save('act_L.npy', act_L)

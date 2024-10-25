from pathlib import Path
from enum import Enum

class DATASET(Enum):
    ISCX=0
    VNAT=1

iscx_map = {
            'email': ['email'],
            'chat': ['aim_chat', 'AIMchat', 'facebook_chat', 'facebookchat', 'hangout_chat', 'hangouts_chat', 'icq_chat', 'ICQchat', 'gmailchat', 'gmail_chat', 'skype_chat'],
            'streaming': ['netflix', 'spotify', 'vimeo', 'youtube', 'youtubeHTML5'],
            'file_transfer': ['ftps_down', 'ftps_up','sftp_up', 'sftpUp', 'sftp_down', 'sftpDown', 'sftp', 'skype_file', 'scpUp', 'scpDown', 'scp'],
            'voip': ['voipbuster', 'facebook_audio', 'hangout_audio', 'hangouts_audio', 'skype_audio'],
            'p2p': ['skype_video', 'facebook_video', 'hangout_video', 'hangouts_video'],

            'vpn_email': ['vpn_email'],
            'vpn_chat': ['vpn_aim_chat', 'vpn_facebook_chat', 'vpn_hangouts_chat', 'vpn_icq_chat', 'vpn_skype_chat'],
            'vpn_streaming': ['vpn_netflix', 'vpn_spotify', 'vpn_vimeo', 'vpn_youtube'],
            'vpn_file_transfer': ['vpn_ftps', 'vpn_sftp', 'vpn_skype_files'],
            'vpn_voip': ['vpn_facebook_audio', 'vpn_skype_audio', 'vpn_voipbuster'],
            'vpn_p2p': ['vpn_bittorrent']   
}

vnat_map = {
            'streaming': ['nonvpn_netflix', 'nonvpn_youtube', 'nonvpn_vimeo'],
            'voip': ['nonvpn_voip', 'nonvpn_skype'],
            'file_transfer': ['nonvpn_rsync', 'nonvpn_sftp', 'nonvpn_scp'],
            'p2p': ['nonvpn_ssh', 'nonvpn_rdp'],

            'vpn_streaming': ['vpn_netflix', 'vpn_youtube', 'vpn_vimeo'],
            'vpn_voip': ['vpn_voip', 'vpn_skype'],
            'vpn_file_transfer': ['vpn_rsync', 'vpn_sftp', 'vpn_scp'],
            'vpn_p2p': ['vpn_ssh', 'vpn_rdp']
}


def iscx_get_unique_labels(): 
    return list(iscx_map.keys())

def vnat_get_unique_labels(): 
    return list(vnat_map.keys())




def iscx_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in iscx_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls


def vnat_get_class_label(file): 
    cls = ''   
    label = file.split('.')[0]
    for m_key, m_values in vnat_map.items():
        for value in m_values:
            if label[:len(value)] == value:
                cls = m_key
                break
    return cls





def iscx_get_one_hot(cls):
    clss = iscx_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot


def vnat_get_one_hot(cls):
    clss = vnat_get_unique_labels()
    index = clss.index(cls)

    one_hot = [0 for _ in range(len(clss))]
    one_hot[index] = 1

    return one_hot


def filenumber_to_id(file_num, length=8):
    file_num_str = str(file_num)
    file_num_str_len = len(file_num_str)
    return '0' * (length - file_num_str_len) + file_num_str

def num_packets_to_edge_indices(num_packets):
    return [list(range(0,num_packets-1)), list(range(1,num_packets))]
    


        








def count_classes(dataset_path, dataset):

    class_names = {}
    if dataset == DATASET.ISCX:
        class_names = {c: 0 for c in list(iscx_map.keys())}
    elif dataset == DATASET.VNAT:
        class_names = {s: 0 for s in list(vnat_map.keys())} 

    # Get list of all PCAP session file paths
    files = list(Path(dataset_path).rglob('*.pcap'))

    for file in enumerate(files, start=1):
        class_label = ''
        if dataset == DATASET.ISCX:
            class_label = iscx_get_class_label(Path(file.__str__()).name)
        elif dataset == DATASET.VNAT:
            class_label = vnat_get_class_label(Path(file.__str__()).name)

        class_names[class_label] = class_names[class_label]  + 1
    
    for key, value in class_names.items():
        print(key, value)
    print('')


        


if __name__=='__main__':
    packets_per_session = 10

    # Process ISCX dataset
    iscx_dataset_path = r'D:\SH\TrafficClassification\vpn-gcn\datasets\ISCX'
    count_classes(iscx_dataset_path, DATASET.ISCX)

    # Process VNAT-VPN dataset
    vnat_dataset_path = r'D:\SH\TrafficClassification\vpn-gcn\datasets\VNAT-VPN'
    count_classes(vnat_dataset_path, DATASET.VNAT)


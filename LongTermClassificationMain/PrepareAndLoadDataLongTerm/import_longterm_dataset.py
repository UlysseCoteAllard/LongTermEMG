import os
import errno
import pickle
import shutil
import pathlib
import xml.etree.ElementTree as ET


def copy(source, destination):
    try:
        shutil.copytree(source, destination)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(source, destination)
        else:
            print('Directory not copied. Error %s' % e)


def get_leap_hand_from_frame(frame):
    arm = {}
    for wristPosition in frame.iter("WristPosition"):
        vector3_position = []
        vector3_position.append(float(wristPosition[0].text))
        vector3_position.append(float(wristPosition[1].text))
        vector3_position.append(float(wristPosition[2].text))
        arm['wristPosition'] = vector3_position

    for arm_xml in frame.iter("Arm"):
        for elbowPosition in arm_xml.iter("PrevJoint"):
            vector3_position = []
            vector3_position.append(float(elbowPosition[0].text))
            vector3_position.append(float(elbowPosition[1].text))
            vector3_position.append(float(elbowPosition[2].text))
            arm['elbowPosition'] = vector3_position

    if len(arm) > 0:  # Make sure that the arm was visible before adding the timestamp
        for timestamp in frame.iter("Timestamp"):
            arm['timestamp'] = timestamp.text

    return arm


def go_over_participants(path_file_ParticipantsDataset, path_file_destination):
    dirs = os.listdir(path_file_ParticipantsDataset)
    for participant_directory in dirs:
        if os.path.isdir(path_file_ParticipantsDataset + "/" + participant_directory):
            print(participant_directory)
            dirs_session = os.listdir(path_file_ParticipantsDataset + "/" + participant_directory)
            for session in dirs_session:
                path_seances = path_file_ParticipantsDataset + "/" + participant_directory + "/" + session
                if os.path.isdir(path_seances):
                    if "Training" in session:
                        print(path_file_destination + "/" + participant_directory + "/" + session)
                        path_seances_armband = path_seances + "/3dc/"
                        print(path_seances_armband)
                        copy(path_seances_armband, path_file_destination + "/" + participant_directory + "/" + session)
                    elif "Evaluation" in session:
                        dirs_files = os.listdir(path_seances)
                        print(dirs_files)
                        for dir_file in dirs_files:
                            if not os.path.isdir(path_seances + "/" + dir_file):
                                print(dir_file)
                                if "Frames" in dir_file:
                                    root = ET.parse(path_seances + "/" + dir_file).getroot()
                                    arm_positions_over_time = []
                                    for frame in root.iter('Frame'):  # Go over all the frames
                                        arm = get_leap_hand_from_frame(frame)
                                        if len(arm) > 0:
                                            arm_positions_over_time.append(arm)
                                    print(arm_positions_over_time)
                                    with open(path_file_destination + "/" + participant_directory + "/" + session +
                                              "/arm_positions.pickle",
                                              'wb') as f:
                                        pickle.dump(arm_positions_over_time, f, pickle.HIGHEST_PROTOCOL)
                                else:
                                    path_seance_evaluations_file = path_file_ParticipantsDataset + "/" +\
                                                                   participant_directory + "/" + session + "/" +\
                                                                   dir_file
                                    print(path_seance_evaluations_file)
                                    pathlib.Path(path_file_destination + "/" + participant_directory + "/" + session
                                                 ).mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(path_seance_evaluations_file,
                                                    path_file_destination + "/" + participant_directory + "/" + session)


if __name__ == "__main__":
    path_file_source = \
        "D:/OneDrive - Université Laval/Doctorat/EMG_VR_Long_Term/Offline_Adaptability/Vr_Software/ParticipantsDataset"
    path_file_destination = "../../datasets/longterm_dataset_3DC"
    print(os.listdir("../../datasets/longterm_dataset_3DC"))
    go_over_participants(path_file_source, path_file_destination)

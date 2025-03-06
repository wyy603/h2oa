import numpy as np

def calculate_euler_angles(joint_positions, joint_names):
    euler_angles = []
    for i, joint_name in enumerate(joint_names):
        if 'x' in joint_name:
            x = joint_positions[i][0]
            y = joint_positions[i][1]
            z = joint_positions[i][2]
            # Calculate Euler angles for x-axis rotation
            roll = np.arctan2(y, x)
            pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
            yaw = 0  # No rotation on y-axis for x-axis rotation
            euler_angles.append((roll, pitch, yaw))
        elif 'y' in joint_name:
            # Calculate Euler angles for y-axis rotation
            x = joint_positions[i][0]
            z = joint_positions[i][2]
            roll = 0  # No rotation on x-axis for y-axis rotation
            pitch = np.arctan2(-z, x)
            yaw = np.arctan2(joint_positions[i][1], np.sqrt(x**2 + z**2))
            euler_angles.append((roll, pitch, yaw))
        elif 'z' in joint_name:
            # Calculate Euler angles for z-axis rotation
            y = joint_positions[i][1]
            z = joint_positions[i][2]
            roll = np.arctan2(-y, z)
            pitch = 0  # No rotation on y-axis for z-axis rotation
            yaw = np.arctan2(joint_positions[i][0], np.sqrt(y**2 + z**2))
            euler_angles.append((roll, pitch, yaw))
        else:
            # Handle other joint names if needed
            pass
    
    return euler_angles

joint_positions = [
    (( 1.12960946e-10, 0.9058    , -8.75000879e-02), (-0.50000006, -0.5000002 ,  0.5000001 ,  0.5000001 )),
    (( 3.92738730e-02, 0.9058    , -9.14103687e-02), (-0.70623654,  0.0350717 ,  0.03507173,  0.7062366 )),
    (( 2.80772969e-02, 0.9289651 , -2.03864604e-01), (-0.4192624 ,  0.47361344,  0.5694025 ,  0.5250623 )),
    (( 2.65739523e-02, 0.53716433, -2.84424156e-01), (-0.42350525,  0.46898165,  0.5732234 ,  0.5216459 )),
    ((-9.15504843e-02, 0.16070852, -3.50211531e-01), (-0.3415279 ,  0.5487485 ,  0.49739733,  0.5786451 )),
    (( 1.79016538e-10, 0.9058    ,  8.74999687e-02), (-0.50000006, -0.5000002 ,  0.5000001 ,  0.5000001 )),
    (( 3.93548645e-02, 0.9058    ,  9.04867128e-02), (-0.7065998 , -0.02677372, -0.02677372,  0.7065997 )),
    (( 3.09933592e-02, 0.938947  ,  2.00665176e-01), (-0.58287966,  0.5509153 ,  0.4003143 ,  0.44327486)),
    (( 7.32638091e-02, 0.55894744,  3.18195730e-01), (-0.61002773,  0.524194  ,  0.4347178 ,  0.40510163)),
    ((-1.53099746e-02, 0.18361163,  4.24393177e-01), (-0.5351942 ,  0.5884846 ,  0.3427024 ,  0.49980837)),
    (( 4.36557457e-11, 1.08      ,  3.50337492e-09), (-0.50000006, -0.5000002 ,  0.5000001 ,  0.5000001 )),
    (( 1.08690359e-01, 1.7731714 , -1.61739103e-02), ( 0.00551762,  0.00261968, -0.9033345 , -0.42889398)),
    (( 1.08873524e-01, 1.7731715 , -3.11728064e-02), ( 0.00551762,  0.00261968, -0.9033345 , -0.42889398)),
    ((-4.47476953e-02, 1.3575602 ,  1.83649026e-02), (-0.70709383, -0.00431777, -0.00431766,  0.7070938 )),
    (( 7.39667751e-03, 1.5099902 , -1.55271381e-01), (-0.38356343,  0.37628037,  0.5940365 ,  0.59867615)),
    (( 5.23158163e-03, 1.5183225 , -2.13757575e-01), (-0.70512116, -0.07726504,  0.02225104,  0.7045135 )),
    (( 2.24917848e-02, 1.3864801 , -1.94885209e-01), (-0.5190661 , -0.51895595,  0.5430883 ,  0.4078123 )),
    (( 6.62709624e-02, 1.1945586 , -1.66675493e-01), (-0.57279426,  0.4921068 ,  0.48812371,  0.43757695)),
    ((-7.21193827e-10, 1.0799999 , -1.11795595e-08), (-0.4969378 ,  0.49693763, -0.50304383,  0.5030439 )),
    (( 4.72963080e-02, 1.7549293 ,  5.77602070e-04), (-0.43227002,  0.5542711 , -0.55959177,  0.43907157)),
    (( 3.60237365e-03, 1.5099902 ,  1.55405685e-01), (-0.5986763 ,  0.59403646,  0.3762804 ,  0.3835633 )),
    ((-2.61278916e-03, 1.5189154 ,  2.13512927e-01), (-0.7070927 , -0.0042651 , -0.00446231,  0.7070939 )),
    ((-2.21042172e-03, 1.3893998 ,  1.77987650e-01), (-0.43151128, -0.5656584 ,  0.42606208,  0.5588383 )),
    (( 1.58434995e-02, 1.196887  ,  1.31521568e-01), (-0.49528673,  0.3499006 ,  0.46081474,  0.64800537))
]

joint_names = ['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 
    'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 
    'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 
    'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 
    'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 
    'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 
    'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 
    'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']


calculate_euler_angles(joint_positions, joint_names)
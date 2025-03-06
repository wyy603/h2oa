from pathlib import Path

# Current file's absolute path
current_file = Path(__file__).resolve()

H2OA_DIR = current_file.parents[1]
SMPL_SIM_DATA = current_file.parents[1] / "retarget" / "SMPLSim" / "smpl_sim" / "data"
LEGGED_GYM_RESOURCES = current_file.parents[1] / "track" / "legged_gym" / "resources"
#DATASET = Path("/cephfs_yili/backup/xuehan/dataset")
DATASET = current_file.parents[1] / "dataset"
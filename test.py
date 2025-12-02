from tqdm import tqdm, trange
from time import sleep

for i in tqdm(range(100), desc="Đang xử lý"):
    sleep(1)
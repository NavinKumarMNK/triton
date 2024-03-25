import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

def download_and_copy_ptxas():
    base_dir = os.path.dirname(__file__) 

    src_path = "bin/ptxas" 
    version = "12.1.105" 
    url = f"https://conda.anaconda.org/nvidia/label/cuda-12.1.1/linux-ppc64le/cuda-nvcc-{version}-0.tar.bz2"  
    dst_prefix = os.path.join(base_dir, "triton") 
    dst_suffix = os.path.join("third_party", "cuda", src_path)
    dst_path = os.path.join(dst_prefix, dst_suffix)  

    is_linux = platform.system() == "Linux"  

    download = False  
    if is_linux: 
        download = True  

        if os.path.exists(dst_path):  
            curr_version = subprocess.check_output([dst_path, "--version"]).decode("utf-8").strip() 
            curr_version = re.search(r"V([.|\d]+)", curr_version).group(1)  
            download = curr_version != version  

    if download:  
        print(f'downloading and extracting {url} ...')
        ftpstream = urllib.request.urlopen(url)  
        file = tarfile.open(fileobj=ftpstream, mode="r|*") 
        with tempfile.TemporaryDirectory() as temp_dir:  
            file.extractall(path=temp_dir)  
            src_path = os.path.join(temp_dir, src_path)  
            os.makedirs(os.path.split(dst_path)[0], exist_ok=True)  
            shutil.copy(src_path, dst_path)  

    return dst_suffix

if __name__ == '__main__':
    download_and_copy_ptxas()

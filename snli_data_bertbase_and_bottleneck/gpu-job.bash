#!/bin/bash

#SBATCH -p defq        # İşin çalıştırılması istenen kuyruk seçilir
#SBATCH -o %j.out      # Çalıştırılan kodun ekran çıktılarını içerir
#SBATCH -e %j.err      # Karşılaşılan hata mesajlarını içerir 
#SBATCH -n 2          # Talep edilen işlemci  çekirdek sayısı
#SBATCH --gres=gpu:1   # Talep edilen GPU sayısı

source activate env_pt

python /okyanus/users/nates/snli_data_bertbase_and_bottleneck/main.py
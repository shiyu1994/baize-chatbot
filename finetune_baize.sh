git clone --recursive https://github.com/project-baize/baize-chatbot.git
cd baize-chatbot
source /opt/conda/etc/profile.d/conda.sh
conda create -n baize 
pip install -r requirements.txt
pip install --upgrade tqdm
python finetune.py 65b 2 0.00005 alpaca,stackoverflow,quora

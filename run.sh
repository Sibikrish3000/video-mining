apt update
apt install -y ffmpeg
apt install -y git
git clone https://github.com/sibikrish3000/video-mining.git
cd video-mining
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sshx.io/get | sh
source $HOME/.local/bin/env
uv sync
curl https://getcroc.schollz.com | bash